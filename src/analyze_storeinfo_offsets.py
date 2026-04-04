"""Verify exact byte offsets in storeinfo 105-byte slot format.

Compares GildyBoye's offsets (+16 item_id, +75 item_id_copy) vs our offsets (+17, +76).
"""

import struct

def main():
    with open(r"F:\projects\crimson-mods\samples\gamedata\storeinfo_raw.bin", "rb") as f:
        data = f.read()

    print(f"File size: {len(data)} bytes")
    print()

    # =========================================================================
    # Part 1: Find Store_Her_Equipment
    # =========================================================================
    target = b"Store_Her_Equipment"
    pos = data.find(target)
    if pos < 0:
        print("ERROR: Store_Her_Equipment not found!")
        return

    print(f"Found 'Store_Her_Equipment' at absolute offset {pos} (0x{pos:X})")

    # Find the null terminator after the name
    name_end = data.index(b'\x00', pos)
    print(f"Name null terminator at offset {name_end} (0x{name_end:X})")
    print(f"Name = '{data[pos:name_end].decode('ascii')}'")
    print(f"Name length (excl null): {name_end - pos}")

    # Read slot_count from name_end + 1 + 37
    slot_count_offset = name_end + 1 + 37
    slot_count = struct.unpack_from('<I', data, slot_count_offset)[0]
    print(f"Slot count at offset {slot_count_offset} (0x{slot_count_offset:X}): {slot_count}")

    # First slot base
    first_slot_base = name_end + 1 + 68
    print(f"First slot base at offset {first_slot_base} (0x{first_slot_base:X})")
    print()

    # =========================================================================
    # Part 2: Dump first 5 slots in detail
    # =========================================================================
    SLOT_SIZE = 105
    num_slots = min(5, slot_count)

    for i in range(num_slots):
        slot_abs = first_slot_base + i * SLOT_SIZE
        slot_data = data[slot_abs:slot_abs + SLOT_SIZE]

        print(f"{'='*80}")
        print(f"SLOT {i} -- absolute offset {slot_abs} (0x{slot_abs:X})")
        print(f"{'='*80}")

        # (a) Print ALL 105 bytes as hex, 10 bytes per line
        for row_start in range(0, SLOT_SIZE, 10):
            row_end = min(row_start + 10, SLOT_SIZE)
            hex_bytes = ' '.join(f'{slot_data[j]:02X}' for j in range(row_start, row_end))
            ascii_repr = ''.join(
                chr(slot_data[j]) if 32 <= slot_data[j] < 127 else '.'
                for j in range(row_start, row_end)
            )
            print(f"  +{row_start:3d}: {hex_bytes:<30s}  |{ascii_repr}|")
        print()

        # (b-i) Read fields at various offsets
        val_0  = struct.unpack_from('<I', slot_data, 0)[0]
        val_16 = struct.unpack_from('<I', slot_data, 16)[0]
        val_17 = struct.unpack_from('<I', slot_data, 17)[0]
        val_75 = struct.unpack_from('<I', slot_data, 75)[0]
        val_76 = struct.unpack_from('<I', slot_data, 76)[0]
        val_90 = struct.unpack_from('<I', slot_data, 90)[0]
        val_98 = struct.unpack_from('<I', slot_data, 98)[0]
        val_104 = slot_data[104]

        print(f"  (b) +0   u32 LE (GildyBoye stock_limit): {val_0} (0x{val_0:08X})")
        print(f"  (c) +16  u32 LE (GildyBoye item_id):     {val_16} (0x{val_16:08X})")
        print(f"  (d) +17  u32 LE (our item_id):            {val_17} (0x{val_17:08X})")
        print(f"  (e) +75  u32 LE (GildyBoye item_id_copy): {val_75} (0x{val_75:08X})")
        print(f"  (f) +76  u32 LE (our item_id_copy):       {val_76} (0x{val_76:08X})")
        print(f"  (g) +90  u32 LE (our buy_price):          {val_90} (0x{val_90:08X})")
        print(f"  (h) +98  u32 LE (our sell_price):         {val_98} (0x{val_98:08X})")
        print(f"  (i) +104 u8    (our is_last):             {val_104} (0x{val_104:02X})")
        print()

        # (7) Verification
        match_gildy = (val_16 == val_75)
        match_ours  = (val_17 == val_76)
        plausible_16 = (1000 <= val_16 <= 10000000)
        plausible_17 = (1000 <= val_17 <= 10000000)

        print(f"  VERIFY: +16 == +75 (GildyBoye pair)?  {'YES' if match_gildy else 'NO'}  ({val_16} vs {val_75})")
        print(f"  VERIFY: +17 == +76 (our pair)?         {'YES' if match_ours else 'NO'}  ({val_17} vs {val_76})")
        print(f"  VERIFY: +16 plausible item_id?         {'YES' if plausible_16 else 'NO'}  (val={val_16})")
        print(f"  VERIFY: +17 plausible item_id?         {'YES' if plausible_17 else 'NO'}  (val={val_17})")

        if match_gildy and not match_ours:
            print(f"  >>> GildyBoye offsets (+16/+75) WIN")
        elif match_ours and not match_gildy:
            print(f"  >>> Our offsets (+17/+76) WIN")
        elif match_gildy and match_ours:
            print(f"  >>> BOTH pairs match -- need more analysis")
        else:
            print(f"  >>> NEITHER pair matches -- something is wrong")

        # (8) Absolute file offsets
        print(f"  Abs offset of +16 field: {slot_abs + 16} (0x{slot_abs + 16:X})")
        print(f"  Abs offset of +17 field: {slot_abs + 17} (0x{slot_abs + 17:X})")
        print()

    # =========================================================================
    # Part 3: Cross-check Store_Her_General
    # =========================================================================
    print(f"{'='*80}")
    print("CROSS-CHECK: Store_Her_General")
    print(f"{'='*80}")

    target2 = b"Store_Her_General"
    pos2 = data.find(target2)
    if pos2 < 0:
        print("ERROR: Store_Her_General not found!")
        return

    print(f"Found 'Store_Her_General' at absolute offset {pos2} (0x{pos2:X})")

    name_end2 = data.index(b'\x00', pos2)
    print(f"Name null terminator at offset {name_end2} (0x{name_end2:X})")

    slot_count_offset2 = name_end2 + 1 + 37
    slot_count2 = struct.unpack_from('<I', data, slot_count_offset2)[0]
    print(f"Slot count: {slot_count2}")

    first_slot_base2 = name_end2 + 1 + 68
    print(f"First slot base: {first_slot_base2} (0x{first_slot_base2:X})")
    print()

    # GildyBoye says absolute offset 108 has item_id hex 71170000 = 6001
    print(f"GildyBoye pre-parsed: slot 1 item_id at abs offset 92+16=108")
    print(f"Reading 4 bytes at absolute offset 108: [{data[108]:02X} {data[109]:02X} {data[110]:02X} {data[111]:02X}]")
    val_at_108 = struct.unpack_from('<I', data, 108)[0]
    print(f"  u32 LE = {val_at_108} (0x{val_at_108:08X})")
    expected_bytes = bytes([0x71, 0x17, 0x00, 0x00])
    print(f"  Expected [71 17 00 00]? {data[108:112] == expected_bytes}")
    print()

    # What's at our computed first slot base?
    print(f"Using computed first_slot_base for Store_Her_General: {first_slot_base2}")
    if first_slot_base2 + SLOT_SIZE <= len(data):
        slot1_data = data[first_slot_base2:first_slot_base2 + SLOT_SIZE]
        # Print first 30 bytes
        for row_start in range(0, 30, 10):
            row_end = min(row_start + 10, SLOT_SIZE)
            hex_bytes = ' '.join(f'{slot1_data[j]:02X}' for j in range(row_start, row_end))
            print(f"  +{row_start:3d}: {hex_bytes}")
        print()

        val_s1_16 = struct.unpack_from('<I', slot1_data, 16)[0]
        val_s1_17 = struct.unpack_from('<I', slot1_data, 17)[0]
        print(f"  +16 u32: {val_s1_16} (0x{val_s1_16:08X})")
        print(f"  +17 u32: {val_s1_17} (0x{val_s1_17:08X})")
        print(f"  Absolute offset of +16: {first_slot_base2 + 16}")
        print(f"  Absolute offset of +17: {first_slot_base2 + 17}")
        print()

    # Check if absolute 108 falls within our computed slot range
    if first_slot_base2 <= 108 < first_slot_base2 + SLOT_SIZE:
        rel = 108 - first_slot_base2
        print(f"  Abs 108 is at relative offset +{rel} within slot 0")
    elif 108 < first_slot_base2:
        print(f"  Abs 108 is {first_slot_base2 - 108} bytes BEFORE first slot base")
        # Check what store record 108 actually falls in
        print(f"  (first_slot_base2={first_slot_base2}, so abs 108 is in the HEADER area)")
    else:
        slot_idx = (108 - first_slot_base2) // SLOT_SIZE
        slot_rel = (108 - first_slot_base2) % SLOT_SIZE
        print(f"  Abs 108 is in slot {slot_idx} at relative offset +{slot_rel}")

    # =========================================================================
    # Part 4: Dump wider context around abs 108
    # =========================================================================
    print()
    print(f"{'='*80}")
    print("CONTEXT: 32 bytes around absolute offset 108")
    print(f"{'='*80}")
    ctx_start = max(0, 108 - 16)
    for row_start in range(ctx_start, min(108 + 16, len(data)), 16):
        row_end = min(row_start + 16, len(data))
        hex_bytes = ' '.join(f'{data[j]:02X}' for j in range(row_start, row_end))
        ascii_repr = ''.join(
            chr(data[j]) if 32 <= data[j] < 127 else '.'
            for j in range(row_start, row_end)
        )
        marker = " <--- 108" if row_start <= 108 < row_end else ""
        print(f"  abs {row_start:5d}: {hex_bytes:<48s}  |{ascii_repr}|{marker}")

    # =========================================================================
    # Part 5: Store_Her_General header analysis
    # =========================================================================
    print()
    print(f"{'='*80}")
    print("Store_Her_General: HEADER bytes (name_end+1 to first_slot_base)")
    print(f"{'='*80}")
    header_start = name_end2 + 1
    for row_start in range(0, first_slot_base2 - header_start, 10):
        row_end = min(row_start + 10, first_slot_base2 - header_start)
        abs_off = header_start + row_start
        hex_bytes = ' '.join(f'{data[abs_off + j]:02X}' for j in range(row_end - row_start))
        print(f"  name_end+1+{row_start:3d} (abs {abs_off:5d}): {hex_bytes}")

    # =========================================================================
    # Part 6: Slot size verification
    # =========================================================================
    print()
    print(f"{'='*80}")
    print("SLOT SIZE VERIFICATION: Try sizes 100-110 for Store_Her_Equipment")
    print(f"{'='*80}")

    for candidate_size in range(100, 111):
        matches_gildy = 0
        matches_ours = 0
        for i in range(min(20, slot_count)):
            slot_off = first_slot_base + i * candidate_size
            if slot_off + candidate_size > len(data):
                break
            sd = data[slot_off:slot_off + candidate_size]
            if len(sd) >= 80:
                v16 = struct.unpack_from('<I', sd, 16)[0]
                v75 = struct.unpack_from('<I', sd, 75)[0]
                v17 = struct.unpack_from('<I', sd, 17)[0]
                v76 = struct.unpack_from('<I', sd, 76)[0]
                if v16 == v75 and 1000 <= v16 <= 10000000:
                    matches_gildy += 1
                if v17 == v76 and 1000 <= v17 <= 10000000:
                    matches_ours += 1
        if matches_gildy > 0 or matches_ours > 0:
            print(f"  Size {candidate_size}: GildyBoye(+16/+75)={matches_gildy}/20, Ours(+17/+76)={matches_ours}/20")

    # =========================================================================
    # Part 7: Brute force -- find consistent offset pairs
    # =========================================================================
    print()
    print(f"{'='*80}")
    print("BRUTE FORCE: slot_size=105, find (a,b) where u32@a==u32@b across 20 slots")
    print(f"{'='*80}")

    best_pairs = []
    for a in range(0, 80):
        for b in range(a + 20, SLOT_SIZE - 3):
            matches = 0
            for i in range(min(20, slot_count)):
                slot_off = first_slot_base + i * SLOT_SIZE
                if slot_off + SLOT_SIZE > len(data):
                    break
                sd = data[slot_off:slot_off + SLOT_SIZE]
                va = struct.unpack_from('<I', sd, a)[0]
                vb = struct.unpack_from('<I', sd, b)[0]
                if va == vb and 1000 <= va <= 10000000:
                    matches += 1
            if matches >= 15:
                best_pairs.append((a, b, matches))

    best_pairs.sort(key=lambda x: -x[2])
    print(f"  Found {len(best_pairs)} pairs with >=15/20 matches:")
    for a, b, m in best_pairs[:30]:
        print(f"    +{a} == +{b}  ({m}/20 matches)")

    # =========================================================================
    # Part 8: Also check GildyBoye's "92+16=108" claim directly
    # =========================================================================
    print()
    print(f"{'='*80}")
    print("GildyBoye absolute offset interpretation")
    print(f"{'='*80}")
    print(f"GildyBoye says Store_Her_General slot 1 starts at abs offset 92.")
    print(f"  This means record_start + header_size = 92")

    # Let's find where Store_Her_General record actually starts
    # The name was found at pos2. The record starts with u16 store_id at pos2 - 6
    rec_start = pos2 - 6
    print(f"  Store_Her_General record starts at abs {rec_start}")
    print(f"  Name at abs {pos2}, name_end at abs {name_end2}")
    print(f"  So GildyBoye's slot 1 start (92) means header = 92 - {rec_start} = {92 - rec_start} from record start")
    print(f"  Our computed first_slot_base = {first_slot_base2} = header of {first_slot_base2 - rec_start} from record start")
    print()

    # What if GildyBoye measures from file start and slot base is just at abs 92?
    print(f"  If GildyBoye's slot 0 is at abs 92:")
    if 92 + SLOT_SIZE <= len(data):
        sd92 = data[92:92 + SLOT_SIZE]
        v16 = struct.unpack_from('<I', sd92, 16)[0]
        v17 = struct.unpack_from('<I', sd92, 17)[0]
        print(f"    +16 u32: {v16} (0x{v16:08X})")
        print(f"    +17 u32: {v17} (0x{v17:08X})")
        print(f"    Is 6001? +16: {v16 == 6001}, +17: {v17 == 6001}")

    # Maybe GildyBoye uses a different header size calculation
    # Let's find where item_id=6001 (0x00001771) appears near Store_Her_General
    print()
    print(f"  Searching for 0x00001771 (=6001 LE) near Store_Her_General...")
    search_bytes = struct.pack('<I', 6001)  # 71 17 00 00
    search_start = max(0, pos2 - 100)
    search_end = min(len(data), pos2 + 2000)
    spos = search_start
    while spos < search_end:
        idx = data.find(search_bytes, spos, search_end)
        if idx == -1:
            break
        print(f"    Found 6001 at abs offset {idx} (0x{idx:X}), rel to name: {idx - pos2}, rel to name_end: {idx - name_end2}")
        # What slot offset would this be?
        if idx >= first_slot_base2:
            slot_num = (idx - first_slot_base2) // SLOT_SIZE
            slot_rel = (idx - first_slot_base2) % SLOT_SIZE
            print(f"      -> slot {slot_num}, relative offset +{slot_rel}")
        spos = idx + 1


if __name__ == "__main__":
    main()
