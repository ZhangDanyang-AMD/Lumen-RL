"""Print the IPv4 address of a network interface (Linux, stdlib only).

Used when the host has no `ip` command; run in a container with --network host.

    python3 netinfo.py <ifname>
"""

import fcntl
import socket
import struct
import sys

SIOCGIFADDR = 0x8915


def main() -> int:
    name = sys.argv[1].encode()[:15]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            raw = fcntl.ioctl(s.fileno(), SIOCGIFADDR, struct.pack("256s", name))
        except OSError as e:
            print(f"no IPv4 address on {sys.argv[1]}: {e}", file=sys.stderr)
            return 1
    print(socket.inet_ntoa(raw[20:24]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
