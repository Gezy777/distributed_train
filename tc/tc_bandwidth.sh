sudo tc qdisc add dev ens5f0np0 root handle 1: htb default 30                   
sudo tc class add dev ens5f0np0 parent 1: classid 1:1 htb rate 2.5Gbit
sudo tc filter add dev ens5f0np0 protocol ip parent 1:0 prio 1 u32 match ip src 0.0.0.0/0 flowid 1:1