import lmdb
import json
import numpy as np
import torch
import msgpack
import msgpack_numpy as m
import os
m.patch()


def move_to_cuda(data):
    """ move data to cuda
    :param data: data to be moved to cuda
    :return:
    """
    if isinstance(data, torch.Tensor):
        return data.cuda(non_blocking=True)

    elif isinstance(data, list):
        data_on_cuda = [move_to_cuda(t) for t in data]

    elif isinstance(data, tuple):
        data_on_cuda = tuple(move_to_cuda(t) for t in data)

    elif isinstance(data, dict):
        data_on_cuda = {n: move_to_cuda(t) for n, t in data.items()}

    else:
        return data

    return data_on_cuda


class LMDBFile(object):
    def __init__(self, data_dir, fea_type, split, tag, encode_method="ndarray", readonly=True):
        f_path = "/%s/%s_%s_%s_lmdb" % (data_dir.strip("/"), split, fea_type, tag)
        create = not readonly
        print(f_path)
        assert os.path.exists(f_path), f_path
        self.env = lmdb.open(f_path, readonly=readonly, create=create, readahead=False, map_size=int(5e10))
        self.txn = self.env.begin()
        self.encode_method = encode_method
        print("LMDB file created: %s" % f_path)

    def put(self, keys, values):
        with self.env.begin(write=True) as txn:
            for k, v in zip(keys, values):
                k_enc, v_enc = self._encode_(k, v)
                txn.put(key=k_enc, value=v_enc, overwrite=True)
            # commit automatically

    def get(self, v_id):
        v_enc = self.txn.get(v_id.encode('utf-8'))
        if v_enc is None:
            return None
        v = self._decode_(v_enc)
        return v

    def _encode_(self, k, v):
        k_enc = k.text_encode()
        if self.encode_method == "ndarray":
            assert isinstance(v, np.ndarray), type(v)
            v_enc = msgpack.packb(v, default=m.encode)

        elif self.encode_method == "json":
            assert isinstance(v, list) or isinstance(v, dict), type(v)
            v_enc = json.dumps(v).encode()

        else:
            assert False, type(v)

        return k_enc, v_enc

    def _decode_(self, v_enc):
        if self.encode_method == "ndarray":
            v = msgpack.unpackb(v_enc, object_hook=m.decode)
            v = np.array(v)
            v = torch.from_numpy(v).float()
        elif self.encode_method == "json":
            v = json.loads(v_enc)
        else:
            assert False, v_enc

        return v

    def keys(self):
        with self.env.begin() as txn:
            keys = [key.decode() for key, _ in txn.cursor()]
        return keys

    def __getitem__(self, video_id):
        if isinstance(video_id, list):
            _feats = [self.get(v_id) for v_id in video_id]
            feats = [i if i is not None else torch.zeros_like(_feats[0]) for i in _feats]
            return torch.stack(feats, dim=0)
        return self.get(video_id)

    def __del__(self):
        self.env.close()
