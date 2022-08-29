# Original code from:
# https://stt.readthedocs.io/en/latest/_downloads/67bac4343abf2261d69231fdaead59fb/client.py

import argparse
import sys
from timeit import default_timer as timer

from transcribe_utils import load_model, load_audio, transcribe_audio


def main():
    parser = argparse.ArgumentParser(description="Running Coqui STT inference.")
    parser.add_argument("--model", required=True, help="Path to the model (protocol buffer binary file)")
    parser.add_argument("--scorer", required=False, help="Path to the external scorer file")
    parser.add_argument("--audio", required=True, help="Path to the audio file to run (WAV format)")
    parser.add_argument("--beam_width", type=int, help="Beam width for the CTC decoder")
    parser.add_argument("--lm_alpha", type=float, help="Language model weight (lm_alpha). If not specified, use default from the scorer package.")
    parser.add_argument("--lm_beta", type=float, help="Word insertion bonus (lm_beta). If not specified, use default from the scorer package.")
    parser.add_argument("--candidate_transcripts", type=int, default=None, help="Number of candidate transcripts to include in JSON output")
    parser.add_argument("--hot_words", type=str, help="Hot-words and their boosts.")
    args = parser.parse_args()

    ds = load_model(args.model, beam_width=args.beam_width,
                    scorer=args.scorer, lm_alpha=args.lm_alpha, lm_beta=args.lm_beta,
                    hot_words=args.hot_words, verbose=True)

    audio, audio_length = load_audio(args.audio, ds.sampleRate())

    print("Running inference.", file=sys.stderr)
    inference_start = timer()
    transcript = transcribe_audio(ds, audio, candidate_transcripts=args.candidate_transcripts)
    print(transcript)
    inference_end = timer() - inference_start
    print("Inference took %0.3fs for %0.3fs audio file." % (inference_end, audio_length), file=sys.stderr)


if __name__ == "__main__":
    main()
