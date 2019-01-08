#!/bin/bash
echo "Attempt to umount data drives of windows computer (/mnt/spts_data and /mnt/spts_backup) ..."
sudo umount -l /mnt/spts_data
sudo umount -l /mnt/spts_backup
