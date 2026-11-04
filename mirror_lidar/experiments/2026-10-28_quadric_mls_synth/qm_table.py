import json, sys
R = json.load(open(sys.argv[1]))['rows']
ms = ['plane', 'sphere', 'quad_free', 'quad_t3', 'quad_t1', 'quad_t0.3']
key = {(r['surf'], r['sigma'], r['h'], r['method']): r for r in R}
for band in ['int', 'edge']:
    print('=== band', band, ' normal err median deg (p95)')
    print(f"{'surf':15s} {'sig':>4s} {'h':>4s} " + ' '.join(f'{m:>12s}' for m in ms))
    for s in dict.fromkeys(r['surf'] for r in R):
        for sig in sorted({r['sigma'] for r in R}):
            for h in sorted({r['h'] for r in R}):
                cells = []
                for m in ms:
                    b = key.get((s, sig, h, m), {}).get(band)
                    cells.append(f"{b['med']:5.2f}({b['p95']:5.1f})" if b else ' ' * 12)
                print(f'{s:15s} {100*sig:4.1f} {h:4.2f} ' + ' '.join(cells))
