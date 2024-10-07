import os
import glob
import numpy
import argparse
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch
from tqdm import tqdm
import torch.nn.functional as F

spk_model = {
    "speechbrain/spkrec-xvect-voxceleb": 512, 
    "speechbrain/spkrec-ecapa-voxceleb": 192,
}

def f2embed(wav_file, classifier, size_embed):
    signal, fs = torchaudio.load(wav_file)
    assert fs == 16000, fs
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        embeddings = F.normalize(embeddings, dim=2)
        embeddings = embeddings.squeeze().cpu().numpy()
    assert embeddings.shape[0] == size_embed, embeddings.shape[0]
    return embeddings

def process(args):
    wavlst = []
    for split in args.splits.split(","):
        wav_dir = os.path.join(args.arctic_root, split)
        wavlst_split = glob.glob(os.path.join(wav_dir, "wav", "*.wav"))
        print(f"{split} {len(wavlst_split)} utterances.")
        wavlst.extend(wavlst_split)

    spkemb_root = args.output_root
    if not os.path.exists(spkemb_root):
        print(f"Create speaker embedding directory: {spkemb_root}")
        os.mkdir(spkemb_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source=args.speaker_embed, run_opts={"device": device}, savedir=os.path.join('/tmp', args.speaker_embed))
    size_embed = spk_model[args.speaker_embed]
    for utt_i in tqdm(wavlst, total=len(wavlst), desc="Extract"):
        # TODO rename speaker embedding
        utt_id = "-".join(utt_i.split("/")[-3:]).replace(".wav", "")
        utt_emb = f2embed(utt_i, classifier, size_embed)
        numpy.save(os.path.join(spkemb_root, f"{utt_id}.npy"), utt_emb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arctic-root", "-i", required=True, type=str, help="LibriTTS root directory.")
    parser.add_argument("--output-root", "-o", required=True, type=str, help="Output directory.")
    parser.add_argument("--speaker-embed", "-s", type=str, required=True, choices=["speechbrain/spkrec-xvect-voxceleb", "speechbrain/spkrec-ecapa-voxceleb"],
                        help="Pretrained model for extracting speaker emebdding.")
    parser.add_argument("--splits",  type=str, help="Split of four speakers seperate by comma.",
                        default="cmu_us_bdl_arctic,cmu_us_clb_arctic,cmu_us_rms_arctic,cmu_us_slt_arctic")
    args = parser.parse_args()
    print(f"Loading utterances from {args.arctic_root}/{args.splits}, "
        + f"Save speaker embedding 'npy' to {args.output_root}, "
        + f"Using speaker model {args.speaker_embed} with {spk_model[args.speaker_embed]} size.")
    process(args)

if __name__ == "__main__":
    """
    python utils/prep_cmu_arctic_spkemb.py \
        -i ~/.cache/huggingface/datasets/facebook___voxpopuli/nl/1.3.0 \
        -o /home/arnav/BTP/FineTuned-SpeakerEmbeddings/dutch \
        -s speechbrain/spkrec-xvect-voxceleb
    """
    main()
