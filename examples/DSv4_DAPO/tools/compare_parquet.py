"""Compare two parquet files by content: row count, schema, and every row in order.

The file bytes depend on the pyarrow/datasets version that wrote them (row-group
layout, metadata), so a sha256 mismatch alone says nothing about the data.

    python3 compare_parquet.py <a.parquet> <b.parquet>
"""

import hashlib
import json
import sys

import pyarrow.parquet as pq


def row_digests(table):
    for batch in table.to_batches(max_chunksize=65536):
        for row in batch.to_pylist():
            yield hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).digest()


def main() -> int:
    a, b = pq.read_table(sys.argv[1]), pq.read_table(sys.argv[2])
    print(f"rows: {a.num_rows} vs {b.num_rows}")
    same_schema = a.schema.remove_metadata().equals(b.schema.remove_metadata())
    print(f"schema equal (ignoring metadata): {same_schema}")
    if not same_schema:
        print(f"  a: {a.schema.remove_metadata()}\n  b: {b.schema.remove_metadata()}")
    da, db = list(row_digests(a)), list(row_digests(b))
    in_order = da == db
    print(f"rows identical and in the same order: {in_order}")
    if not in_order:
        sa, sb = set(da), set(db)
        first = next((i for i, (x, y) in enumerate(zip(da, db)) if x != y), min(len(da), len(db)))
        print(f"  first differing row index: {first}")
        print(f"  distinct rows only in a: {len(sa - sb)}, only in b: {len(sb - sa)}")
        print(f"  same multiset of rows: {sorted(da) == sorted(db)}")
    return 0 if (same_schema and in_order) else 1


if __name__ == "__main__":
    raise SystemExit(main())
