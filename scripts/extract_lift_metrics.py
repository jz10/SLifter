#!/usr/bin/env python3
import re
import csv
from pathlib import Path


BENCH_DIR = Path('bench_outputs')
TEST_ROOT = Path('test/hecbench')
SM = '75'


def parse_type_counts(text: str):
    # Find the last TypeAnalysis block
    start_tok = '=== Start of TypeAnalysis ==='
    end_tok = '=== End of TypeAnalysis ==='
    last_start = text.rfind(start_tok)
    last_end = text.rfind(end_tok)
    if last_start == -1 or last_end == -1 or last_end < last_start:
        return {}
    block = text[last_start:last_end]
    counts = {}
    for m in re.finditer(r'^Type counts\s+([A-Za-z0-9_]+):\s+(\d+)\s+registers', block, re.M):
        typ, n = m.group(1), int(m.group(2))
        # If duplicate keys in the same block, keep the last occurrence
        counts[typ] = n
    return counts


def parse_patterns(text: str):
    # Defaults
    res = {
        'pack64_total': 0,
        'xmad_to_imad_p1': 0,
        'xmad_to_imad_p2': 0,
        'fphack_p1': 0,
        'operagg_cast64_total': 0,
        'operagg_insert_total': 0,
        'operagg_remove_total': 0,
    }

    m = re.search(r'Total PACK64 instructions added:\s*(\d+)', text)
    if m:
        res['pack64_total'] = int(m.group(1))

    # xmad -> imad patterns (pattern 1 and 2). Be lenient about whitespace.
    m = re.search(r'Transformed\s+(\d+)\s+set of xmad instructions\s*\(pattern\s*1\)\s*to imad instructions\.', text, re.S)
    if m:
        res['xmad_to_imad_p1'] = int(m.group(1))
    m = re.search(r'Transformed\s+(\d+)\s+set of xmad instructions\s*\(pattern\s*2\)\s*to imad instructions\.', text, re.S)
    if m:
        res['xmad_to_imad_p2'] = int(m.group(1))

    m = re.search(r'FPHack:\s*replaced\s*(\d+)\s*pattern1', text)
    if m:
        res['fphack_p1'] = int(m.group(1))

    # Prefer Total OperAggregate ... if present, else sum non-total per-block entries.
    m_cast = re.search(r'Total OperAggregate Cast64 Inserts:\s*(\d+)', text)
    m_ins = re.search(r'Total OperAggregate Insert Instructions:\s*(\d+)', text)
    m_rem = re.search(r'Total OperAggregate Remove Instructions:\s*(\d+)', text)
    if m_cast and m_ins and m_rem:
        res['operagg_cast64_total'] = int(m_cast.group(1))
        res['operagg_insert_total'] = int(m_ins.group(1))
        res['operagg_remove_total'] = int(m_rem.group(1))
    else:
        # Sum plain OperAggregate lines
        res['operagg_cast64_total'] = sum(int(x) for x in re.findall(r'OperAggregate Cast64 Inserts:\s*(\d+)', text))
        res['operagg_insert_total'] = sum(int(x) for x in re.findall(r'OperAggregate Insert Instructions:\s*(\d+)', text))
        res['operagg_remove_total'] = sum(int(x) for x in re.findall(r'OperAggregate Remove Instructions:\s*(\d+)', text))

    return res


def count_sass_instruction_lines(sass_path: Path) -> int:
    if not sass_path.exists():
        return 0
    n = 0
    patt = re.compile(r'^\s*/\*\s*[0-9a-fA-F]{4}\s*\*/')
    with sass_path.open('r', errors='ignore') as f:
        for line in f:
            if patt.search(line):
                n += 1
    return n


def count_nonempty_lines(p: Path) -> int:
    if not p.exists():
        return 0
    with p.open('r', errors='ignore') as f:
        return sum(1 for line in f if line.strip())


def parse_srsubstitute_operands(text: str) -> int:
    m = re.search(r'SRSubstitute: processed\s+(\d+)\s+operands', text)
    return int(m.group(1)) if m else 0


def parse_mov_elim_removed(text: str) -> int:
    m = re.search(r'MovEliminate: removed\s+(\d+)\s+mov instructions', text)
    return int(m.group(1)) if m else 0


def ir_instruction_counts(ll_path: Path):
    c = {k: 0 for k in ('phi', 'load', 'store', 'br')}
    if not ll_path.exists():
        return c
    with ll_path.open('r', errors='ignore') as f:
        for line in f:
            s = line.strip()
            # Count exact instruction keywords
            if re.search(r'\bphi\b', s):
                c['phi'] += 1
            if re.search(r'^\s*load\b', line):
                c['load'] += 1
            if re.search(r'^\s*store\b', line):
                c['store'] += 1
            if re.search(r'^\s*br\b', line):
                c['br'] += 1
    return c


def sass_memory_counts(sass_path: Path):
    c = {k: 0 for k in ('LDG', 'STG', 'LDS', 'STS')}
    if not sass_path.exists():
        return c
    patt = re.compile(r'\b(LDG|STG|LDS|STS)\b')
    with sass_path.open('r', errors='ignore') as f:
        for line in f:
            for m in patt.finditer(line):
                c[m.group(1)] += 1
    return c


def parse_type_conflicts_resolved(text: str) -> int:
    # Count how many times TypeAnalysis inserted a BITCAST to resolve conflicts
    return len(re.findall(r'Warning: Inserting BITCAST to resolve type conflict', text))


def main():
    logs = sorted(BENCH_DIR.glob('*.lift.log'))
    if not logs:
        print('No .lift.log files found in bench_outputs/')
        return

    # Collect per-benchmark data
    type_rows = {}
    pattern_rows = {}
    lines_rows = {}
    extra_rows = {}
    all_types = set()

    for log_path in logs:
        bench = log_path.stem.replace('.lift', '')  # e.g., 'blockexchange'
        text = log_path.read_text(errors='ignore')

        # Types
        type_counts = parse_type_counts(text)
        type_rows[bench] = type_counts
        all_types.update(type_counts.keys())

        # Patterns
        pattern_rows[bench] = parse_patterns(text)

        # Lines (SASS vs IR)
        sass_path = TEST_ROOT / f'{bench}-cuda' / f'{bench}_sm{SM}.sass'
        ll_path = BENCH_DIR / f'{bench}.ll'
        lines_rows[bench] = {
            'sass_inst_lines': count_sass_instruction_lines(sass_path),
            'llvm_ir_nonempty_lines': count_nonempty_lines(ll_path),
        }

        # Extras: SRSubstitute, MovEliminate, IR instruction counts, SASS mem ops
        extras = {
            'srsubstitute_operands': parse_srsubstitute_operands(text),
            'mov_eliminate_removed': parse_mov_elim_removed(text),
            'type_conflicts_resolved': parse_type_conflicts_resolved(text),
        }
        extras.update({f'ir_{k}': v for k, v in ir_instruction_counts(ll_path).items()})
        extras.update({f'sass_{k.lower()}': v for k, v in sass_memory_counts(sass_path).items()})
        extra_rows[bench] = extras

    # Write CSVs
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    # 1) types.csv
    types_csv = BENCH_DIR / 'types.csv'
    fieldnames = ['benchmark'] + sorted(all_types)
    with types_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for bench in sorted(type_rows.keys()):
            row = {'benchmark': bench}
            for t in all_types:
                row[t] = type_rows[bench].get(t, 0)
            w.writerow(row)

    # 2) patterns.csv
    patterns_csv = BENCH_DIR / 'patterns.csv'
    patt_fields = [
        'benchmark',
        'pack64_total',
        'xmad_to_imad_p1',
        'xmad_to_imad_p2',
        'fphack_p1',
        'operagg_cast64_total',
        'operagg_insert_total',
        'operagg_remove_total',
    ]
    with patterns_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=patt_fields)
        w.writeheader()
        for bench in sorted(pattern_rows.keys()):
            row = {'benchmark': bench}
            row.update(pattern_rows[bench])
            w.writerow(row)

    # 3) lines.csv
    lines_csv = BENCH_DIR / 'lines.csv'
    line_fields = ['benchmark', 'sass_inst_lines', 'llvm_ir_nonempty_lines']
    with lines_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=line_fields)
        w.writeheader()
        for bench in sorted(lines_rows.keys()):
            row = {'benchmark': bench}
            row.update(lines_rows[bench])
            w.writerow(row)

    # 4) extras.csv
    extras_csv = BENCH_DIR / 'extras.csv'
    extra_fields = [
        'benchmark',
        'srsubstitute_operands', 'mov_eliminate_removed', 'type_conflicts_resolved',
        'ir_phi', 'ir_load', 'ir_store', 'ir_br',
        'sass_ldg', 'sass_stg', 'sass_lds', 'sass_sts',
    ]
    with extras_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=extra_fields)
        w.writeheader()
        for bench in sorted(extra_rows.keys()):
            row = {'benchmark': bench}
            row.update(extra_rows[bench])
            w.writerow(row)

    print(f'Wrote {types_csv}, {patterns_csv}, {lines_csv}')


if __name__ == '__main__':
    main()
