import os
import subprocess

mtime=0
f=open('my_out.txt','a+')
f.write("recording_started\n")
f.close()
while(True):
	mountpoint = "/var/lib/docker/volumes/world/_data"
	saved_file = mountpoint+'./last'
	new_mtime = os.path.getmtime(saved_file)
	if mtime!=new_mtime:
		last=open(saved_file)
		msgs = last.readlines()
		f=open('my_out.txt','a')
		for msg in msgs:
			f.write(msg)
			f.write('\n')
		f.close()