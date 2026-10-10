"""Memory-mapped token sequences: a `<prefix>.bin` data file and a `<prefix>.idx` index.

The format is fairseq/Megatron's MMIDIDX: a header (magic, version, dtype
code, sequence count, document count) followed by int32 sizes, int64 byte
pointers and int64 document indices.
"""

import struct

import numpy as np

MAGIC = b"MMIDIDX\x00\x00"
DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64,
          6: np.float32, 7: np.double, 8: np.uint16, 9: np.uint32}
DTYPE_CODES = {dtype: code for code, dtype in DTYPES.items()}


class IndexedTokensWriter:
    def __init__(self, prefix, dtype):
        self.prefix = prefix
        self.dtype = dtype
        self.sizes = []
        self.data = open(prefix + ".bin", "wb")

    def add(self, tokens):
        array = np.asarray(tokens, dtype=self.dtype)
        self.data.write(array.tobytes(order="C"))
        self.sizes.append(array.size)

    def finalize(self):
        self.data.close()
        sizes = np.array(self.sizes, dtype=np.int32)
        itemsize = np.dtype(self.dtype).itemsize
        pointers = np.zeros(len(sizes), dtype=np.int64)
        pointers[1:] = np.cumsum(sizes[:-1], dtype=np.int64) * itemsize
        doc_idx = np.array([0], dtype=np.int64)
        with open(self.prefix + ".idx", "wb") as index:
            index.write(MAGIC)
            index.write(struct.pack("<Q", 1))
            index.write(struct.pack("<B", DTYPE_CODES[self.dtype]))
            index.write(struct.pack("<Q", len(sizes)))
            index.write(struct.pack("<Q", len(doc_idx)))
            index.write(sizes.tobytes(order="C"))
            index.write(pointers.tobytes(order="C"))
            index.write(doc_idx.tobytes(order="C"))


class IndexedTokens:
    """Read-only random access to the sequences of `<prefix>.bin/.idx`."""

    def __init__(self, prefix):
        self.prefix = prefix
        with open(prefix + ".idx", "rb") as index:
            if index.read(9) != MAGIC:
                raise ValueError(f"{prefix}.idx is not an MMIDIDX index")
            if struct.unpack("<Q", index.read(8)) != (1,):
                raise ValueError(f"{prefix}.idx has an unsupported version")
            (code,) = struct.unpack("<B", index.read(1))
            (count,) = struct.unpack("<Q", index.read(8))
            index.read(8)  # document count, unused
            offset = index.tell()
        self.dtype = DTYPES[code]
        index_buffer = memoryview(np.memmap(prefix + ".idx", mode="r", order="C"))
        self.sizes = np.frombuffer(index_buffer, dtype=np.int32, count=count, offset=offset)
        self.pointers = np.frombuffer(index_buffer, dtype=np.int64, count=count,
                                      offset=offset + self.sizes.nbytes)
        self.data = memoryview(np.memmap(prefix + ".bin", mode="r", order="C"))

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, i):
        return np.frombuffer(self.data, dtype=self.dtype, count=self.sizes[i], offset=self.pointers[i])

    # DataLoader workers started with spawn re-open the memory maps.
    def __getstate__(self):
        return self.prefix

    def __setstate__(self, prefix):
        self.__init__(prefix)
