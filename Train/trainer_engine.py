import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

import logging
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING) 

import tqdm
import os
import sys
import torch.multiprocessing as mp
import json # Añadir para el estado de Gradio

import src
from src import utils
from src.data_utils import TextAudioCollate, TextAudioLoader, DistributedBucketSampler
from src.Voice_Synthesizer import AVAILABLE_DURATION_DISCRIMINATOR_TYPES, AVAILABLE_FLOW_TYPES
from src.Modules import DurationDiscriminatorV2, DurationDiscriminatorV1
from src.Voice_Synthesizer import SynthesizerTrn

from src.Text.Symbols import symbols
from src.Audio import spec_to_mel_torch,mel_spectrogram_torch
from src.losses import generator_loss, discriminator_loss, feature_loss, kl_loss

from  src.Modules.Period_Discriminators import MultiPeriodDiscriminator
from src.Modules import commons







class Trainer:
    def __init__(self, rank, n_gpus, hps):
           # --- 1. Configuración del Entorno y Proceso ---
        self.rank = rank
        self.n_gpus = n_gpus
        self.hps = hps
        self.global_step = 0
        self.epoch = 1

        torch.manual_seed(hps.train.seed)
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="gloo", init_method="env://", world_size=n_gpus, rank=rank)


        if (
        "use_mel_posterior_encoder" in hps.model.keys()
        and hps.model.use_mel_posterior_encoder == True
        ):
            print("Using mel posterior encoder for VITS2")
            posterior_channels = 80  # vits2
            hps.data.use_mel_posterior_encoder = True
        else:
            print("Using lin posterior encoder for VITS1")
            posterior_channels = hps.data.filter_length // 2 + 1
            hps.data.use_mel_posterior_encoder = False
    

        if rank == 0:
            self.logger = utils.get_logger(hps.model_dir)
            self.writer = SummaryWriter(log_dir=hps.model_dir)
            self.writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
            self.logger.info(hps)
            utils.check_git_hash(hps.model_dir)

        # --- 2. Carga de Datos ---
        self.train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
        train_sampler = DistributedBucketSampler(
            self.train_dataset, hps.train.batch_size, [32, 300, 400, 500, 600, 700, 800, 900, 1000],
            num_replicas=n_gpus, rank=rank, shuffle=True
        )
        collate_fn = TextAudioCollate()
        self.train_loader = DataLoader(
            self.train_dataset, num_workers=8, shuffle=False, pin_memory=True,
            collate_fn=collate_fn, batch_sampler=train_sampler
        )
        if rank == 0:
            self.eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
            self.eval_loader = DataLoader(
                self.eval_dataset, num_workers=8, shuffle=False, batch_size=hps.train.batch_size,
                pin_memory=True, drop_last=False, collate_fn=collate_fn
            )

        # --- 3. Inicialización de Modelos ---
        self.net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        mas_noise_scale_initial=0.01,
        noise_scale_delta=2e-6,
        **hps.model,
        ).cuda(rank)
        self.net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
        # Aquí puedes agregar la lógica para el discriminador de duración si es necesario
        if (
        "use_duration_discriminator" in hps.model.keys()
        and hps.model.use_duration_discriminator == True
        ):
            # print("Using duration discriminator for VITS2")
            use_duration_discriminator = True
            duration_discriminator_type = hps.model.duration_discriminator_type
            print(f"Using duration_discriminator {duration_discriminator_type} for VITS2")
            assert duration_discriminator_type in AVAILABLE_DURATION_DISCRIMINATOR_TYPES, f"duration_discriminator_type must be one of {AVAILABLE_DURATION_DISCRIMINATOR_TYPES}"
            if duration_discriminator_type == "dur_disc_1":
                self.net_dur_disc = DurationDiscriminatorV1(
                    hps.model.hidden_channels,
                    hps.model.hidden_channels,
                    3,
                    0.1,
                    gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
                ).cuda(rank)
            elif duration_discriminator_type == "dur_disc_2":
                self.net_dur_disc = DurationDiscriminatorV2(
                    hps.model.hidden_channels,
                    hps.model.hidden_channels,
                    3,
                    0.1,
                    gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
                ).cuda(rank) 
        else:
            print("NOT using any duration discriminator like VITS1")
            self.net_dur_disc = None
            use_duration_discriminator = False

        # --- 4. Envoltura DDP ---

        # Siempre usamos DDP porque mp.spawn crea un entorno distribuido
        self.net_g = DDP(self.net_g, device_ids=[rank], find_unused_parameters=True)
        self.net_d = DDP(self.net_d, device_ids=[rank], find_unused_parameters=True)
        if self.net_dur_disc is not None:
            self.net_dur_disc = DDP(self.net_dur_disc, device_ids=[rank], find_unused_parameters=True)

        if self.hps.train.fp16_run:
            self.net_g.half()
            self.net_d.half()
            if self.net_dur_disc is not None:
                self.net_dur_disc.half()

        # --- 5. Optimizadores y Schedulers ---
        self.optim_g = torch.optim.AdamW(self.net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
        self.optim_d = torch.optim.AdamW(self.net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
        if self.net_dur_disc is not None:
            self.optim_dur_disc = torch.optim.AdamW(
                self.net_dur_disc.parameters(),
                hps.train.learning_rate,
                betas=hps.train.betas,
                eps=hps.train.eps,
            )
        else:
            self.optim_dur_disc = None
    
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=hps.train.lr_decay)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=hps.train.lr_decay)
        if self.net_dur_disc is not None:
            self.scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
                    self.optim_dur_disc, gamma=hps.train.lr_decay)
        else:
            self.scheduler_dur_disc = None


        # --- 6. Carga de Checkpoint ---
        try:
            _, _, _, self.epoch = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), self.net_g, self.optim_g)
            _, _, _, self.epoch =  utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), self.net_d, self.optim_d)
            if self.net_dur_disc is not None:
                _, _, _, self.epoch = utils.load_checkpoint(
                    utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                    self.net_dur_disc,
                    self.optim_dur_disc,
                )
            self.global_step = (self.epoch - 1) * len(self.train_loader)
            self.scheduler_g.last_epoch = self.epoch - 2
            self.scheduler_d.last_epoch = self.epoch - 2
            if self.net_dur_disc is not None:
                self.scheduler_dur_disc.last_epoch = self.epoch - 2
            else:
                self.scheduler_dur_disc = None
        except:
            self.epoch = 1
            self.global_step = 0
        
        # --- 7. Scaler para FP16 ---
        self.scaler = GradScaler(enabled=hps.train.fp16_run)

    def train(self):
        """ Bucle de entrenamiento principal que itera sobre las épocas. """
        for epoch in range(self.epoch, self.hps.train.epochs + 1):
            self.epoch = epoch
            self._train_one_epoch()
            
            self.scheduler_g.step()
            self.scheduler_d.step()
            if self.scheduler_dur_disc is not None:
                self.scheduler_dur_disc.step()

    def _train_one_epoch(self):
        """
        Lógica para entrenar una sola época completa.
        """
        # Establece la época para el sampler distribuido, asegurando una mezcla adecuada de los datos
        self.train_loader.batch_sampler.set_epoch(self.epoch)
        
        # Pone los modelos en modo de entrenamiento
        self.net_g.train()
        self.net_d.train()
        if self.net_dur_disc is not None:
            self.net_dur_disc.train()

        # Configura la barra de progreso TQDM solo en el proceso principal (rank 0)
        loader = self.train_loader
        if self.rank == 0:
            loader = tqdm.tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        # Itera sobre cada batch del dataset de entrenamiento
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(loader):
            
            # Lógica para "Noise Scaled Monotonic Alignment Search" (MAS)
            if getattr(self.net_g.module, 'use_noise_scaled_mas', False):
                current_mas_noise_scale = (
                    self.net_g.module.mas_noise_scale_initial
                    - self.net_g.module.noise_scale_delta * self.global_step
                )
                self.net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)

            # Mueve los datos a la GPU correspondiente
            x, x_lengths = x.cuda(self.rank, non_blocking=True), x_lengths.cuda(self.rank, non_blocking=True)
            spec, spec_lengths = spec.cuda(self.rank, non_blocking=True), spec_lengths.cuda(self.rank, non_blocking=True)
            y, y_lengths = y.cuda(self.rank, non_blocking=True), y_lengths.cuda(self.rank, non_blocking=True)

            # --- 1. ENTRENAMIENTO DEL DISCRIMINADOR ---
            with autocast(enabled=self.hps.train.fp16_run):
                # El generador produce audio falso (y_hat)
                y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (hidden_x, logw, logw_) = self.net_g(x, x_lengths, spec, spec_lengths)
                if (
                self.hps.model.use_mel_posterior_encoder
                or self.hps.data.use_mel_posterior_encoder
                ):
                    mel = spec
                else:
                    mel = spec_to_mel_torch(
                        spec.float(),
                        self.hps.data.filter_length,
                        self.hps.data.n_mel_channels,
                        self.hps.data.sampling_rate,
                        self.hps.data.mel_fmin,
                        self.hps.data.mel_fmax,
                    )
                y_mel = commons.slice_segments(
                                mel, ids_slice, self.hps.train.segment_size // self.hps.data.hop_length
                            )
                y_hat_mel = mel_spectrogram_torch(
                                y_hat.squeeze(1),
                                self.hps.data.filter_length,
                                self.hps.data.n_mel_channels,
                                self.hps.data.sampling_rate,
                                self.hps.data.hop_length,
                                self.hps.data.win_length,
                                self.hps.data.mel_fmin,
                                self.hps.data.mel_fmax,
                            )

                y = commons.slice_segments(
                                y, ids_slice * self.hps.data.hop_length, self.hps.train.segment_size
                            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())
            with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
                    loss_disc_all = loss_disc

                # Duration Discriminator
            if self.net_dur_disc is not None:
                    y_dur_hat_r, y_dur_hat_g = self.net_dur_disc(
                        hidden_x.detach(), x_mask.detach(), logw_.detach(), logw.detach()
                    )
                    with autocast(enabled=False):
                        # TODO: I think need to mean using the mask, but for now, just mean all
                        (
                            loss_dur_disc,
                            losses_dur_disc_r,
                            losses_dur_disc_g,
                        ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                        loss_dur_disc_all = loss_dur_disc
                    self.optim_dur_disc.zero_grad()
                    self.scaler.scale(loss_dur_disc_all).backward()
                    self.scaler.unscale_(self.optim_dur_disc)
                    grad_norm_dur_disc = commons.clip_grad_value_(
                        self.net_dur_disc.parameters(), None
                    )
                    self.scaler.step(self.optim_dur_disc)

            self.optim_d.zero_grad()
            self.scaler.scale(loss_disc_all).backward()
            self.scaler.unscale_(self.optim_d)
            grad_norm_d = commons.clip_grad_value_(self.net_d.parameters(), None)
            self.scaler.step(self.optim_d)

            with autocast(enabled=self.hps.train.fp16_run):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
                if self.net_dur_disc is not None:
                    y_dur_hat_r, y_dur_hat_g = self.net_dur_disc(hidden_x, x_mask, logw_, logw)
                with autocast(enabled=False):
                    loss_dur = torch.sum(l_length.float())
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hps.train.c_kl

                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                    if self.net_dur_disc is not None:
                        loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                        loss_gen_all += loss_dur_gen

            self.optim_g.zero_grad()
            self.scaler.scale(loss_gen_all).backward()
            self.scaler.unscale_(self.optim_g)
            grad_norm_g = commons.clip_grad_value_(self.net_g.parameters(), None)
            self.scaler.step(self.optim_g)
            self.scaler.update()

            # --- 4. LOGGING Y ESCRITURA DE ESTADO ---
            if self.rank == 0 and self.global_step % self.hps.train.log_interval == 0:
                lr = self.optim_g.param_groups[0]['lr']
                
                # Log a la consola
                self.logger.info(f"Epoch: {self.epoch} [{100. * batch_idx / len(self.train_loader):.1f}%], Step: {self.global_step}, Loss G: {loss_gen_all:.4f}, Loss D: {loss_disc:.4f}")
                
                # Log a TensorBoard
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                    "loss/g/fm": loss_fm,
                    "loss/g/mel": loss_mel,
                    "loss/g/dur": loss_dur,
                    "loss/g/kl": loss_kl,
                }
                if self.net_dur_disc is not None:
                    scalar_dict['loss/dur_disc/total'] = loss_dur_disc
                    scalar_dict['loss/g/dur_gen'] = loss_dur_gen


                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/attn": utils.plot_alignment_to_numpy(
                        attn[0, 0].data.cpu().numpy()
                    ),
                }

                utils.summarize(writer=self.writer, global_step=self.global_step, scalars=scalar_dict,images=image_dict)
                
                # Creación y guardado del archivo de estado para Gradio
                status_dict = { "epoch": self.epoch, "total_epochs": self.hps.train.epochs, "step": self.global_step, "loss_g": f"{loss_gen_all:.4f}", "loss_d": f"{loss_disc:.4f}" }
                status_path = os.path.join(self.hps.model_dir, "status.json")
                with open(status_path, 'w') as f:
                    json.dump(status_dict, f)

            self.global_step += 1
            if self.rank == 0 and self.global_step % self.hps.train.eval_interval == 0:
                self._evaluate()
                print(f"DEBUG: Guardando checkpoint en el paso {self.global_step}...")
                utils.save_checkpoint(self.net_g, self.optim_g, self.hps.train.learning_rate, self.epoch, os.path.join(self.hps.model_dir, f"G_{self.global_step}.pth"))
                utils.save_checkpoint(self.net_d, self.optim_d, self.hps.train.learning_rate, self.epoch, os.path.join(self.hps.model_dir, f"D_{self.global_step}.pth"))
                if self.net_dur_disc is not None:
                    utils.save_checkpoint(self.net_dur_disc, self.optim_dur_disc, self.hps.train.learning_rate, self.epoch, os.path.join(self.hps.model_dir, f"DUR_{self.global_step}.pth"))
                utils.remove_old_checkpoints(self.hps.model_dir, prefixes=["G_*.pth", "D_*.pth", "DUR_*.pth"])

    def _evaluate(self):
            """
            Evalúa el modelo con un ejemplo del dataset de validación.
            """
            if self.rank == 0:
                self.net_g.eval()
                with torch.no_grad():
                    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(
                        self.eval_loader
                    ):
                        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
                        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
                        y, y_lengths = y.cuda(0), y_lengths.cuda(0)

                        # remove else
                        x = x[:1]
                        x_lengths = x_lengths[:1]
                        spec = spec[:1]
                        spec_lengths = spec_lengths[:1]
                        y = y[:1]
                        y_lengths = y_lengths[:1]
                        break
                    y_hat, attn, mask, *_ = self.net_g.module.infer(x, x_lengths, max_len=1000)
                    y_hat_lengths = mask.sum([1, 2]).long() * self.hps.data.hop_length

                    if self.hps.model.use_mel_posterior_encoder or self.hps.data.use_mel_posterior_encoder:
                        mel = spec
                    else:
                        mel = spec_to_mel_torch(
                            spec,
                            self.hps.data.filter_length,
                            self.hps.data.n_mel_channels,
                            self.hps.data.sampling_rate,
                            self.hps.data.mel_fmin,
                            self.hps.data.mel_fmax,
                        )
                    y_hat_mel = mel_spectrogram_torch(
                        y_hat.squeeze(1).float(),
                        self.hps.data.filter_length,
                        self.hps.data.n_mel_channels,
                        self.hps.data.sampling_rate,
                        self.hps.data.hop_length,
                        self.hps.data.win_length,
                        self.hps.data.mel_fmin,
                        self.hps.data.mel_fmax,
                    )
                image_dict = {
                    "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
                }
                audio_dict = {"gen/audio": y_hat[0, :, : y_hat_lengths[0]]}
                if self.global_step == 0:
                    image_dict.update(
                        {"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())}
                    )
                    audio_dict.update({"gt/audio": y[0, :, : y_lengths[0]]})

                utils.summarize(
                    writer=self.writer_eval,
                    global_step=self.global_step,
                    images=image_dict,
                    audios=audio_dict,
                    audio_sampling_rate=self.hps.data.sampling_rate,
                )
                self.net_g.train()

# --- PUNTO DE ENTRADA ---

def start_worker(rank, n_gpus, hps):
    """Función objetivo que cada proceso de GPU ejecutará."""
    trainer = Trainer(rank, n_gpus, hps)
    trainer.train()

if __name__ == "__main__":
    assert torch.cuda.is_available(), "El entrenamiento en CPU no está permitido."
    
    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6060" # O cualquier otro puerto libre
    
    hps = utils.get_hparams()
    

    mp.spawn(
        start_worker,
        nprocs=n_gpus,
        args=(n_gpus, hps,)
    )