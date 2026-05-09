#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rewrite Cooja .csc client mote type while preserving mote positions.

The script updates only the client motetype's <source> and <commands>, leaving
all <mote> entries (IDs + positions) unchanged for rigorous A/B comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def pretty_indent(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            pretty_indent(e, level + 1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def rewrite_client_type(
    input_csc: Path,
    output_csc: Path,
    client_source: str,
    build_target: str,
) -> None:
    tree = ET.parse(input_csc)
    root = tree.getroot()

    sim = root.find("simulation")
    if sim is None:
        raise ValueError("Invalid .csc: missing <simulation>.")

    motetypes = sim.findall("motetype")
    if len(motetypes) < 2:
        raise ValueError("Expected at least 2 motetypes (server + client).")

    client_mt = None
    for mt in motetypes:
        src = mt.find("source")
        if src is None or src.text is None:
            continue
        if "udp-client" in src.text:
            client_mt = mt
            break
    if client_mt is None:
        client_mt = motetypes[-1]

    src_elem = client_mt.find("source")
    cmd_elem = client_mt.find("commands")
    if src_elem is None or cmd_elem is None:
        raise ValueError("Client motetype missing <source> or <commands>.")

    src_elem.text = f"[CONTIKI_DIR]/examples/rpl-udp/{client_source}"
    cmd_elem.text = f"$(MAKE) -j$(CPUS) {build_target}.cooja TARGET=cooja"

    pretty_indent(root)
    output_csc.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_csc, encoding="UTF-8", xml_declaration=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rewrite client motetype in Cooja .csc")
    ap.add_argument("--input_csc", required=True)
    ap.add_argument("--output_csc", required=True)
    ap.add_argument("--client_source", required=True, help="e.g. udp-client-mix-ldp.c")
    ap.add_argument("--build_target", required=True, help="e.g. udp-client-mix-ldp")
    args = ap.parse_args()

    rewrite_client_type(
        input_csc=Path(args.input_csc),
        output_csc=Path(args.output_csc),
        client_source=str(args.client_source),
        build_target=str(args.build_target),
    )
    print(f"[OK] wrote {args.output_csc}")


if __name__ == "__main__":
    main()
