def load_transcripts(files):
    transcripts = {}
    with open(files, 'r', encoding='utf-8') as f:
        for line in f:
            segments = line.strip().split(' | ')
            transcripts[segments[0]] = segments[1]
    return transcripts


def correct_transcripts(main_file, reprocessed_file, correct_file):
    main_transcripts = load_transcripts(main_file)
    reprocessed_transcripts = load_transcripts(reprocessed_file)

    new_transcripts = main_transcripts.copy()  # Do a copy

    for files, text in reprocessed_transcripts.items():
        if files in main_transcripts:
            new_transcripts[files] = text

    with open(correct_file, 'w', encoding='utf-8') as f:
        for files, text in new_transcripts.items():
            f.write(f"{files} | {text}\n")


main_file = "/content/wavs_text_original.txt"
reprocessed_file = "/content/wavs_text_reprocessed.txt"
correct_file = "wavs_text_correct.txt"

correct_transcripts(main_file, reprocessed_file, correct_file)