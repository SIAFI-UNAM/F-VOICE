import argparse
import text
from tqdm import tqdm
from utils import load_filepaths_and_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_extension", default="cleaned")
    parser.add_argument("--text_index", default=1, type=int)
    parser.add_argument(
        "--filelists",
        nargs="+",
        default=[
            "filelists/ljs_audio_text_val_filelist.txt",
            "filelists/ljs_audio_text_test_filelist.txt",
        ],
    )
    parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])

    args = parser.parse_args()

    for filelist in args.filelists:
            print(f"\n🔹 Procesando archivo: {filelist}")
            
            filepaths_and_text = load_filepaths_and_text(filelist)
            total_lines = len(filepaths_and_text)

            for i in tqdm(range(total_lines), desc=f"📜 Limpiando {filelist}", unit=" línea"):
                original_text = filepaths_and_text[i][args.text_index]
                cleaned_text = text._clean_text(original_text, args.text_cleaners)
                filepaths_and_text[i][args.text_index] = cleaned_text

            # Generar nombre del nuevo archivo
            new_filelist = f"{filelist}.{args.out_extension}"

            # Guardar el archivo limpio
            with open(new_filelist, "w", encoding="utf-8") as f:
                f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])

            print(f"✅ Archivo procesado y guardado como: {new_filelist}")