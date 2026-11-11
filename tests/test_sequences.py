import torch

from src.cache import cache_from_snapshot
from src.sequences import pack_thoughts, pack_token_rows, position_ids


def test_pack_keeps_the_first_eos_as_the_loop_token():
    prompt_ids = torch.tensor([[0, 0, 11, 12], [21, 22, 23, 24]])
    prompt_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    sequences = torch.tensor(
        [
            [0, 0, 11, 12, 31, 99, 0],
            [21, 22, 23, 24, 41, 42, 99],
        ]
    )
    prefix_ids, prefix_mask, last_ids, last_pos = pack_thoughts(prompt_ids, prompt_mask, sequences, eos_id=99)
    assert last_ids.view(-1).tolist() == [99, 99]
    assert last_pos.tolist() == [3, 6]
    assert prefix_mask[0].tolist() == [0, 0, 0, 1, 1, 1]
    assert prefix_ids[0].tolist()[-3:] == [11, 12, 31]
    assert prefix_ids[1].tolist() == [21, 22, 23, 24, 41, 42]
    positions = position_ids(prefix_mask)
    assert positions[0].tolist()[-3:] == [0, 1, 2]
    assert int(positions[0, 0]) == 0


def test_pack_without_eos_uses_the_final_token():
    prompt_ids = torch.tensor([[5, 6]])
    prompt_mask = torch.tensor([[1, 1]])
    sequences = torch.tensor([[5, 6, 7, 8]])
    prefix_ids, prefix_mask, last_ids, last_pos = pack_thoughts(prompt_ids, prompt_mask, sequences, eos_id=99)
    assert prefix_ids.tolist() == [[5, 6, 7]]
    assert prefix_mask.tolist() == [[1, 1, 1]]
    assert int(last_ids) == 8
    assert int(last_pos) == 3


def test_stored_rows_loop_the_final_token():
    prefix_ids, prefix_mask, last_ids, last_pos = pack_token_rows([[11, 12, 31, 99], [21, 22, 23, 24, 41, 42, 99]])
    assert last_ids.view(-1).tolist() == [99, 99]
    assert last_pos.tolist() == [3, 6]
    assert prefix_ids[1].tolist() == [21, 22, 23, 24, 41, 42]
    assert prefix_mask[0].tolist() == [0, 0, 0, 1, 1, 1]


def test_snapshot_cache_keeps_prefix_length():
    pairs = [(torch.zeros(2, 2, 5, 4), torch.ones(2, 2, 5, 4))]
    cache = cache_from_snapshot(pairs)
    assert cache.get_seq_length() == 5
    again = cache_from_snapshot(pairs)
    assert again.get_seq_length() == 5
    assert cache.layers[0].keys.data_ptr() != again.layers[0].keys.data_ptr()
