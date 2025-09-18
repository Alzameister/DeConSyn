import base64
import hashlib
import io
import struct
import zlib
from typing import Dict, Any, Union, Tuple

import torch


def _pack(sd: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    torch.save(sd, buf, _use_new_zipfile_serialization=False)
    return zlib.compress(buf.getvalue(), level=3)

def _unpack(comp: bytes) -> Dict[str, Any]:
    obj = torch.load(io.BytesIO(zlib.decompress(comp)), map_location="cpu")
    for k, v in list(obj.items()):
        if torch.is_tensor(v) and v.dtype == torch.float64:
            obj[k] = v.float()
    return obj

def encode_state_dict_pair_blob(
    gen_sd: Dict[str, Any],
    dis_sd: Dict[str, Any],
    as_ascii: bool = True,
) -> Union[str, bytes]:
    g = _pack(gen_sd)
    d = _pack(dis_sd)
    header = struct.pack("<QQ", len(g), len(d))
    blob = header + g + d
    return base64.b85encode(blob).decode("ascii") if as_ascii else blob

def decode_state_dict_pair_blob(
    blob: Union[str, bytes],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if isinstance(blob, str):
        blob = base64.b85decode(blob.encode("ascii"))
    if len(blob) < 16:
        raise ValueError("Blob too small")
    gen_len, dis_len = struct.unpack_from("<QQ", blob, 0)
    off = 16
    if len(blob) != 16 + gen_len + dis_len:
        raise ValueError("Length mismatch")
    g = blob[off:off+gen_len]; off += gen_len
    d = blob[off:off+dis_len]
    return _unpack(g), _unpack(d)


def encode_state_dict_pair(gen_sd: Dict[str, Any], dis_sd: Dict[str, Any]) -> Dict[str, Any]:
    gen_comp = _pack(gen_sd)
    dis_comp = _pack(dis_sd)
    checksum = hashlib.sha256(gen_comp + dis_comp).hexdigest()
    return {
        "generator": base64.b64encode(gen_comp).decode("utf-8"),
        "discriminator": base64.b64encode(dis_comp).decode("utf-8"),
        "checksum": checksum,
        "gen_bytes": len(gen_comp),
        "dis_bytes": len(dis_comp),
    }

def decode_state_dict_pair(package: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    gen_comp = base64.b64decode(package["generator"])
    dis_comp = base64.b64decode(package["discriminator"])
    checksum = package["checksum"]
    if hashlib.sha256(gen_comp + dis_comp).hexdigest() != checksum:
        raise ValueError("Checksum mismatch.")
    gen_sd = _unpack(gen_comp)
    dis_sd = _unpack(dis_comp)
    return gen_sd, dis_sd