"""Evaluate a saved CRNN checkpoint on the complete synthetic tier splits."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from data.collate import talim_collate_fn
from data.encoder import CTCLabelEncoder
from data.tier_dataset import TalimTierDataset
from data.vocabulary import TalimVocabulary
from models.crnn import CRNN
from training.train_crnn import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', choices=['validation', 'test'], default=['validation', 'test'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    mapping = checkpoint['char_to_idx']
    vocabulary = TalimVocabulary.for_tiers(mapping)
    vocabulary.char_to_index = mapping
    vocabulary.index_to_char = {i: c for c, i in mapping.items()}
    vocabulary.index_to_char[0] = vocabulary.blank_token
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(num_classes=len(vocabulary)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    results = {'checkpoint': str(args.checkpoint), 'checkpoint_global_step': checkpoint.get('global_step'),
               'num_classes': len(vocabulary), 'scope': 'held-out synthetic tiers; not real-handwriting accuracy'}
    for split in args.splits:
        loader = DataLoader(TalimTierDataset(args.dataset_root / split), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, collate_fn=talim_collate_fn)
        results['full_' + split] = evaluate(model, loader, CTCLabelEncoder(vocabulary),
                                           torch.nn.CTCLoss(blank=0, zero_infinity=True), vocabulary,
                                           device, split, validate_ctc_runtime=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
