#!/bin/bash
echo "Attempt to umount data drives of windows computer (/mnt/msi_data and /mnt/msi_backup) ..."
sudo umount -l /mnt/msi_data
sudo umount -l /mnt/msi_backup
