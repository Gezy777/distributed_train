tc qdisc show dev ens5f0np0
sudo tc qdisc add dev ens5f0np0 root netem delay 10ms
tc qdisc show dev ens5f0np0