#!/usr/bin/env python

import os

IP_WIN="130.238.39.21"
with open("/etc/rc.local_msi.secret") as f:
    lines = f.readlines()
    username = lines[0][:-1]
    password = lines[1][:-1]

print "INFO: Attempt to mount data drives of windows computer under /mnt/msi_data and /mnt/msi_backup ..."

err1 = os.system("mount -t cifs -o username=%s,pass=%s //130.238.39.21/E /mnt/msi_data" % (username,password))
err2 = os.system("mount -t cifs -o username=%s,pass=%s //130.238.39.21/F /mnt/msi_backup" % (username,password))

if (err1 == 0) and (err2 == 0):
    print "INFO: Successfully mounted /mnt/msi_data and /mnt/msi_backup"
else:
    print "ERROR: Successfully mounted /mnt/msi_data and /mnt/msi_backup"



