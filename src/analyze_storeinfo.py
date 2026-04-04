"""Analyze storeinfo.pabgb binary format — full decode of Hernand equipment store.

FINDINGS:
- storeinfo_raw.bin has NO pabgb markers (73 e1 c5 ea). It's raw record data.
- 290 store records, no index — just sequential records.
- Each record: [u16 store_id][u32 name_len][name][null][variable header][N x 105-byte item entries]
- Item entries are exactly 105 bytes, with '01 00 01 01' marker at entry+13.
- Header size varies by store type (67, 115, 172, 220, 277, 382, 592 bytes after name).
- The last item entry has byte+104 = 1 (is_last flag).
"""

import struct
import sys
import os

# ── Load raw data ────────────────────────────────────────────────

RAW_PATH = r"F:\projects\crimson-mods\samples\gamedata\storeinfo_raw.bin"
data = open(RAW_PATH, 'rb').read()
print(f"storeinfo_raw.bin: {len(data):,} bytes\n")

# ── Load item database for name resolution ───────────────────────

sys.path.insert(0, r"F:\projects\crimson-mods\lazorr410-unpacker\python")
sys.path.insert(0, r"F:\projects\crimson-mods\tools\src")
game_dir = r"F:\programs\steam\steamapps\common\Crimson Desert"
from item_db import parse_localization, parse_iteminfo
print("Loading item database...")
loc = parse_localization(game_dir)
items = parse_iteminfo(game_dir, loc)
id_to_item = {it.item_id: it for it in items}
print(f"  {len(loc):,} loc strings, {len(items):,} items\n")

# ── Find all store records ───────────────────────────────────────

ENTRY_SIZE = 105
ENTRY_MARKER = bytes([0x01, 0x00, 0x01, 0x01])  # at entry+13
MARKER_OFFSET = 13

stores = []
pos = 0
while pos < len(data) - 6:
    idx = data.find(b'Store_', pos)
    if idx == -1:
        break
    if idx < 6:
        pos = idx + 1
        continue
    name_len = struct.unpack_from('<I', data, idx - 4)[0]
    store_id = struct.unpack_from('<H', data, idx - 6)[0]
    if name_len < 6 or name_len > 60 or idx + name_len > len(data):
        pos = idx + 1
        continue
    name_bytes = data[idx:idx + name_len]
    if not all(0x20 <= b < 0x7f for b in name_bytes):
        pos = idx + 1
        continue
    name = name_bytes.decode('ascii')
    stores.append((idx - 6, store_id, name, name_len))
    pos = idx + name_len + 1

# Calculate record sizes and find items
store_records = []
for i in range(len(stores)):
    rs, sid, nm, nlen = stores[i]
    next_rs = stores[i + 1][0] if i + 1 < len(stores) else len(data)
    rec_size = next_rs - rs
    # Count items by finding 105-byte-spaced marker chains
    srec = data[rs:rs + rec_size]
    markers = []
    p = 0
    while True:
        mi = srec.find(ENTRY_MARKER, p)
        if mi == -1:
            break
        markers.append(mi)
        p = mi + 1
    # Build the longest chain of 105-apart markers
    if markers:
        chain = [markers[0]]
        for m in markers[1:]:
            if m - chain[-1] == ENTRY_SIZE:
                chain.append(m)
        n_items = len(chain)
        first_entry_off = chain[0] - MARKER_OFFSET
        header_size = first_entry_off
    else:
        n_items = 0
        header_size = rec_size
    store_records.append((rs, sid, nm, nlen, rec_size, n_items, header_size))

print(f"Total stores: {len(store_records)}\n")

# ── Find Store_Her_Equipment ─────────────────────────────────────

target = None
for sr in store_records:
    if sr[2] == "Store_Her_Equipment":
        target = sr
        break

rs, sid, nm, nlen, rsz, n_items, hdr_sz = target
rec = data[rs:rs + rsz]

print("=" * 90)
print(f"STORE: {nm}  (id={sid}, offset=0x{rs:06x}, size={rsz})")
print(f"  Header: {hdr_sz} bytes, Items: {n_items}, Entry size: {ENTRY_SIZE}")
print(f"  Verify: {hdr_sz} + {n_items}*{ENTRY_SIZE} = {hdr_sz + n_items * ENTRY_SIZE} == {rsz}")
print("=" * 90)

# ── Extract entries ──────────────────────────────────────────────

entries = []
for i in range(n_items):
    entries.append(rec[hdr_sz + i * ENTRY_SIZE: hdr_sz + (i + 1) * ENTRY_SIZE])


# ── HEADER HEX DUMP ─────────────────────────────────────────────

print(f"\nHEADER ({hdr_sz} bytes):")
print("-" * 90)
hdr = rec[:hdr_sz]
for i in range(0, len(hdr), 16):
    chunk = hdr[i:min(i + 16, len(hdr))]
    hexs = ' '.join(f'{b:02x}' for b in chunk)
    hexs = hexs.ljust(48)
    ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
    print(f"  {i:04x}: {hexs} |{ascii_str}|")

# ── FULL HEX DUMP of entire record ──────────────────────────────

print(f"\nFULL RECORD HEX DUMP ({rsz} bytes):")
print("-" * 90)
for i in range(0, rsz, 16):
    chunk = rec[i:min(i + 16, rsz)]
    hexs = ' '.join(f'{b:02x}' for b in chunk)
    hexs = hexs.ljust(48)
    ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
    annot = ""
    if i < hdr_sz:
        annot = " [HEADER]"
    else:
        entry_idx = (i - hdr_sz) // ENTRY_SIZE
        entry_off = (i - hdr_sz) % ENTRY_SIZE
        if entry_idx < n_items:
            if entry_off < 16:
                annot = f" [ENTRY {entry_idx}]"
    print(f"  {i:04x}: {hexs} |{ascii_str}|{annot}")


# ── ITEM ENTRY STRUCTURE ────────────────────────────────────────

print()
print("=" * 90)
print("ITEM ENTRY BINARY STRUCTURE (105 bytes):")
print("=" * 90)
print("""
  Offset  Size  Type     Field                    Notes
  ------  ----  ----     -----                    -----
  +0      1     u8       separator                Always 0
  +1      1     u8       sort_priority            Higher = displayed first (60,40,10,4,3,2,1)
  +2      3     u8[3]    padding                  Zeros
  +5      1     u8       display_order            Sequential: 2,3,4,...,N+1
  +6      3     u8[3]    padding                  Zeros
  +9      1     u8       slot_id                  Unique slot in store (may jump between categories)
  +10     3     u8[3]    padding                  Zeros
  +13     4     bytes    marker                   Constant: 01 00 01 01
  +17     4     u32      item_id                  References iteminfo.pabgb
  +21     13    u8[13]   reserved                 Usually zeros
  +34     4     u32      hash_a                   Item-specific hash (or 0)
  +38     8     u8[8]    reserved                 Usually zeros
  +46     4     u32      hash_b                   Item-specific hash (or 0)
  +50     8     u8[8]    reserved                 Usually zeros
  +58     1     u8       active_flag              1 = active, 0 = some items
  +59     13    u8[13]   reserved                 Zeros
  +72     2     u8[2]    padding                  Zeros
  +74     2     u16      separator                Always 0xFFFF
  +76     4     u32      item_id_copy             Same as +17 (redundant copy)
  +80     4     i32      variant_code             0 or negative (e.g. -868, -870)
  +84     4     u8[4]    padding                  Zeros
  +88     1     u8       price_flag               1 = standard pricing, 0 = special
  +89     1     u8       padding                  Zero
  +90     4     u32      buy_price                Price to buy (e.g. 1,000,000 silver)
  +94     4     u8[4]    padding                  Zeros
  +98     4     u32      sell_price               Price to sell back
  +102    2     u8[2]    padding                  Zeros
  +104    1     u8       is_last                  1 if last entry in store, else 0
""")


# ── FIRST 5 ENTRIES — ANNOTATED ─────────────────────────────────

print("=" * 90)
print(f"FIRST 5 ITEM ENTRIES (of {n_items}):")
print("=" * 90)

for idx in range(min(5, n_items)):
    e = entries[idx]
    iid = struct.unpack_from('<I', e, 17)[0]
    iid_copy = struct.unpack_from('<I', e, 76)[0]
    buy = struct.unpack_from('<I', e, 90)[0]
    sell = struct.unpack_from('<I', e, 98)[0]
    variant = struct.unpack_from('<i', e, 80)[0]
    hash_a = struct.unpack_from('<I', e, 34)[0]
    hash_b = struct.unpack_from('<I', e, 46)[0]

    rec_item = id_to_item.get(iid)
    dn = rec_item.display_name if rec_item else "(not in iteminfo)"
    iname = rec_item.internal_name if rec_item else ""

    print(f"\n--- Entry {idx} (rec_off=0x{hdr_sz + idx * ENTRY_SIZE:04x}) ---")
    # Full hex
    for i in range(0, ENTRY_SIZE, 16):
        chunk = e[i:min(i + 16, ENTRY_SIZE)]
        hexs = ' '.join(f'{b:02x}' for b in chunk)
        hexs = hexs.ljust(48)
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        print(f"  +{i:04x}: {hexs} |{ascii_str}|")
    print(f"  Decoded:")
    print(f"    sort_priority  = {e[1]}")
    print(f"    display_order  = {e[5]}")
    print(f"    slot_id        = {e[9]}")
    print(f"    item_id        = {iid} ({iname})")
    print(f"    display_name   = \"{dn}\"")
    print(f"    hash_a         = 0x{hash_a:08x}" if hash_a else "    hash_a         = 0")
    print(f"    hash_b         = 0x{hash_b:08x}" if hash_b else "    hash_b         = 0")
    print(f"    active_flag    = {e[58]}")
    print(f"    item_id_copy   = {iid_copy}")
    print(f"    variant_code   = {variant}")
    print(f"    price_flag     = {e[88]}")
    print(f"    buy_price      = {buy:,}")
    print(f"    sell_price     = {sell:,}")
    print(f"    is_last        = {e[104]}")


# ── ALL 23 ITEMS WITH NAMES ─────────────────────────────────────

print()
print("=" * 90)
print(f"ALL {n_items} ITEMS IN Store_Her_Equipment:")
print("=" * 90)
print(f"{'#':>3s} {'item_id':>10s} {'prio':>4s} {'ord':>3s} {'slot':>4s} "
      f"{'variant':>8s} {'buy':>10s} {'last':>4s}  {'display_name'}")
print("-" * 90)

for idx in range(n_items):
    e = entries[idx]
    iid = struct.unpack_from('<I', e, 17)[0]
    buy = struct.unpack_from('<I', e, 90)[0]
    variant = struct.unpack_from('<i', e, 80)[0]
    rec_item = id_to_item.get(iid)
    dn = rec_item.display_name if rec_item else "(unknown)"
    print(f"{idx:3d} {iid:10d} {e[1]:4d} {e[5]:3d} {e[9]:4d} "
          f"{variant:8d} {buy:10,} {e[104]:4d}  {dn}")


# ── ALL HERNAND STORES ───────────────────────────────────────────

print()
print("=" * 90)
print("ALL HERNAND (Her) STORES:")
print("=" * 90)
for sr in store_records:
    if "Her" in sr[2]:
        print(f"  id={sr[1]:5d}  items={sr[5]:3d}  size={sr[4]:6d}  \"{sr[2]}\"")


# ── CRAFTING/MATERIAL STORES ────────────────────────────────────

print()
print("=" * 90)
print("STORES POTENTIALLY SELLING CRAFTING MATERIALS:")
print("=" * 90)
for sr in store_records:
    low = sr[2].lower()
    if any(kw in low for kw in ['general', 'material', 'craft', 'resource',
                                  'leather', 'stone', 'blacksmith', 'grocery',
                                  'nonghyeop', 'ore']):
        print(f"  id={sr[1]:5d}  items={sr[5]:3d}  size={sr[4]:6d}  \"{sr[2]}\"")


# ── EQUIPMENT STORE CROSS-CHECK ─────────────────────────────────

print()
print("=" * 90)
print("ALL EQUIPMENT STORES (entry_size=105 verification):")
print("=" * 90)
for sr in store_records:
    if "Equipment" in sr[2]:
        rs, sid, nm, nlen, rsz, n, hsz = sr
        expected = hsz + n * ENTRY_SIZE
        status = "OK" if expected == rsz else f"TAIL={rsz - expected}"
        print(f"  {nm:40s} id={sid:5d} items={n:3d} hdr={hsz:3d} {status}")


# ── HEADER FIELD DECODE ─────────────────────────────────────────

print()
print("=" * 90)
print("HEADER FIELD DECODE (Store_Her_Equipment):")
print("=" * 90)
off = 0
print(f"  +{off:3d}: u16 store_id = {struct.unpack_from('<H', hdr, off)[0]}")
off = 2
print(f"  +{off:3d}: u32 name_len = {struct.unpack_from('<I', hdr, off)[0]}")
off = 6
print(f"  +{off:3d}: str name = \"{hdr[off:off+nlen].decode('ascii')}\"")
off = 6 + nlen + 1
print(f"  +{off:3d}: --- header body ({hdr_sz - off} bytes) ---")
for j in range(off, hdr_sz, 4):
    if j + 4 <= hdr_sz:
        val = struct.unpack_from('<I', hdr, j)[0]
        ival = struct.unpack_from('<i', hdr, j)[0]
        bx = hdr[j:j + 4].hex()
        extra = ""
        if val == 1000000:
            extra = "  (1,000,000 - default price)"
        elif val == 1 and ival == 1:
            extra = "  (=1)"
        elif ival == -1:
            extra = "  (=-1, sentinel)"
        print(f"  +{j:3d}: {bx}  u32={val:11d}  i32={ival:11d}{extra}")


# ── BYTE-BY-BYTE MAP OF ONE COMPLETE ENTRY ──────────────────────

print()
print("=" * 90)
print("BYTE-BY-BYTE MAP: Entry #4 (item_id=720001, has non-zero hashes + variant)")
print("=" * 90)

e = entries[4]
field_map = [
    (0, 1, "separator", "u8"),
    (1, 1, "sort_priority", "u8"),
    (2, 3, "padding", "bytes"),
    (5, 1, "display_order", "u8"),
    (6, 3, "padding", "bytes"),
    (9, 1, "slot_id", "u8"),
    (10, 3, "padding", "bytes"),
    (13, 1, "marker[0]", "u8"),
    (14, 1, "marker[1]", "u8"),
    (15, 1, "marker[2]", "u8"),
    (16, 1, "marker[3]", "u8"),
    (17, 4, "item_id", "u32"),
    (21, 13, "reserved_a", "bytes"),
    (34, 4, "hash_a", "u32"),
    (38, 8, "reserved_b", "bytes"),
    (46, 4, "hash_b", "u32"),
    (50, 8, "reserved_c", "bytes"),
    (58, 1, "active_flag", "u8"),
    (59, 13, "reserved_d", "bytes"),
    (72, 2, "padding_e", "bytes"),
    (74, 2, "separator_ffff", "u16"),
    (76, 4, "item_id_copy", "u32"),
    (80, 4, "variant_code", "i32"),
    (84, 4, "padding_f", "bytes"),
    (88, 1, "price_flag", "u8"),
    (89, 1, "padding_g", "bytes"),
    (90, 4, "buy_price", "u32"),
    (94, 4, "padding_h", "bytes"),
    (98, 4, "sell_price", "u32"),
    (102, 2, "padding_i", "bytes"),
    (104, 1, "is_last", "u8"),
]

for off, sz, name, dtype in field_map:
    raw = e[off:off + sz]
    hexs = raw.hex(' ')
    if dtype == "u8":
        val = raw[0]
        print(f"  +{off:3d} [{sz:2d}B]: {hexs:12s}  {name:20s} = {val}")
    elif dtype == "u16":
        val = struct.unpack_from('<H', raw, 0)[0]
        print(f"  +{off:3d} [{sz:2d}B]: {hexs:12s}  {name:20s} = {val} (0x{val:04x})")
    elif dtype == "u32":
        val = struct.unpack_from('<I', raw, 0)[0]
        print(f"  +{off:3d} [{sz:2d}B]: {hexs:12s}  {name:20s} = {val} (0x{val:08x})")
    elif dtype == "i32":
        val = struct.unpack_from('<i', raw, 0)[0]
        print(f"  +{off:3d} [{sz:2d}B]: {hexs:12s}  {name:20s} = {val}")
    elif dtype == "bytes":
        print(f"  +{off:3d} [{sz:2d}B]: {hexs:12s}  {name:20s}")

print(f"\n  Total: {sum(sz for _, sz, _, _ in field_map)} bytes (should be {ENTRY_SIZE})")
