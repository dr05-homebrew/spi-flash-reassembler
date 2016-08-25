#!/usr/bin/env python2
from __future__ import division
import os
import sys
import re
import zipfile
import json
import ConfigParser
import numpy as np

class Tee(object):
	def __init__(self, name, mode='w'):
		self.file = open(name, mode)
		self.stdout = sys.stdout
		#sys.stdout = self
		self.linestart = 0

	def __del__(self):
		#sys.stdout = self.stdout
		self.file.close()

	def write(self, data):
		for ch in data:
			if ch == '\r':
				self.file.seek(self.linestart)
			else:
				self.file.write(ch)
				if ch == '\n':
					self.linestart = self.file.tell()

		self.stdout.write(data)

si_prefixes = {
	'k': 3,
	'M': 6,
	'G': 9,
}

def parse_quantity(text, unit=None):
	if unit is None:
		m = re.match(r'^([0-9.+-]+)\s?(\w)(\w*)$', text)
		(_num, _prefix, _unit) = m.groups()
	else:
		assert text.endswith(unit)
		text = text[:-len(unit)]
		m = re.match(r'^([0-9.+-]+)\s?(\w)$', text)
		(_num, _prefix) = m.groups()

	num = float(_num)
	exp = si_prefixes.get(_prefix, _prefix)

	if unit is None:
		return (num * 10**exp, _unit)
	else:
		return (num * 10**exp)

def dict_select(dict, *alternatives, **kws):
	reverse = kws.get('reverse', False)
	default = kws.get('default', None)

	if reverse:
		dict = {
			(dict[k].lower() if isinstance(dict[k], str) else dict[k]): k
			for k in dict
		}
	else:
		dict = {
			(k.lower() if isinstance(k, str) else k): dict[k]
			for k in dict
		}

	for alt in alternatives:
		alt = alt.lower()
		if alt in dict:
			return dict[alt]

	return default

class SigrokFile(object):
	def __init__(self, filename):
		file = self.file = zipfile.ZipFile(filename)

		config = self.config = ConfigParser.ConfigParser()
		config.readfp(file.open('metadata'))

		assert 'device 1' in config.sections()

		totalprobes = config.getint('device 1', 'total probes')
		self.samplerate = parse_quantity(config.get('device 1', 'samplerate'), 'Hz')
		self.capturefile = config.get('device 1', 'capturefile')

		self.probes = {}
		for k in xrange(totalprobes):
			probekey = "probe{}".format(k+1)
			if config.has_option('device 1', probekey):
				self.probes[k] = config.get('device 1', probekey)

		self.numchunks = len(list(self))

		self.duration = self.numsamples / self.samplerate

		assert totalprobes <= 8 # or else need to use bigger sample type

	def __iter__(self):
		try:
			self.file.getinfo(self.capturefile)
			yield self.capturefile
			return # done
		except KeyError:
			pass # multiple chunks

		counter = 0
		offset = 0
		while True:
			counter += 1
			chunkname = "{}-{}".format(self.capturefile, counter)

			try:
				self.file.getinfo(chunkname)
			except KeyError:
				raise StopIteration

			yield chunkname

	@property
	def numsamples(self):
		return sum(
			self.file.getinfo(chunkname).file_size
			for chunkname in self)

	def readchunks(self):
		offset = 0
		for chunkname in self:
			try:
				self.file.getinfo(chunkname)
			except KeyError:
				raise StopIteration

			chunk = np.fromstring(
				self.file.open(chunkname).read(),
				dtype=np.uint8)
			
			yield offset, chunk

			offset += len(chunk)


class SPIFlash(object):
	def __init__(self, nbytes=None, sectorsize=2**12, blocksize=2**16):
		self.check_consistency = True
		self.sectorsize = sectorsize
		self.blocksize = blocksize
		self.nbytes = 0

		if nbytes is None:
			nbytes = 1 * self.blocksize
		else:
			assert nbytes % self.blocksize == 0

		self.data = np.zeros(0, dtype=np.uint8)
		self.mask = np.zeros(0, dtype=np.bool)

		self.resize(nbytes)

	def __len__(self):
		assert self.nbytes == len(self.data)
		return self.nbytes

	def resize(self, newbytes):
		assert newbytes % self.blocksize == 0

		overlap = min(newbytes, self.nbytes)

		self.nbytes = newbytes

		newdata = np.zeros(newbytes, dtype=np.uint8)
		newmask = np.zeros(newbytes, dtype=np.bool)

		newdata[:overlap] = self.data[:overlap]
		newmask[:overlap] = self.mask[:overlap]

		self.data = newdata
		self.mask = newmask


	@classmethod
	def fromfile(cls, srcfile):
		nbytes = os.path.getsize(srcfile)
		result = cls(nbytes=nbytes)
		result.data = np.fromfile(srcfile, dtype=np.uint8)
		result.mask[:] = True

	def save(self, dstfile):
		self.data.tofile(dstfile)

	def save_mask(self, dstfile):
		self.mask.tofile(dstfile)

	def load_mask(self, srcfile):
		self.mask = np.fromfile(srcfile, dtype=np.bool)

	def erase(self, addr, length):
		assert length > 0
		assert length % 512 == 0
		assert addr % length == 0
		self.mask[addr:addr+length] = False
		self.data[addr:addr+length] = 0xFF
		print
		print "erase {:06X} + {:d}".format(addr, length)

	def write(self, addr, data):
		length = len(data)

		while addr + length >= len(self):
			self.resize(len(self) * 2)

		if self.check_consistency:
			overlap = self.mask[addr:addr+length]
			#assert not ((self.data[addr:addr+length] != data) & overlap).any()
		self.mask[addr:addr+length] = True
		self.data[addr:addr+length] = data

########################################################################

# spi:miso=0:mosi=1:clk=2:cs=3

infile = sys.argv[1]

sys.stdout = Tee(infile + ".log")

source = SigrokFile(infile)
freq = source.samplerate
print "sample rate: {:.6f} MHz".format(freq / 1e6)
print "probes:", json.dumps(source.probes, indent=1, sort_keys=True)

ch_miso = dict_select(source.probes, 'miso', default=0, reverse=True)
ch_mosi = dict_select(source.probes, 'mosi', default=1, reverse=True)
ch_sclk = dict_select(source.probes, 'sclk', 'clk', default=2, reverse=True)
ch_cs   = dict_select(source.probes, 'cs', 'cs#', '/cs', default=3, reverse=True)

assert ch_cs is not None, "Probe for CS must be labeled"

# TODO: command-line setting

# cs active-low
# sample on rising edge
# msb first

nchannels = max(source.probes) + 1

nsamples = source.numsamples

signal = np.zeros((nsamples, nchannels), dtype=np.int8)

for i, (offset, chunk) in enumerate(source.readchunks()):
	sys.stdout.write("\rloading part {}/{}, {:.1f}%, at {:.3f}s, {:.1f}M samples".format(
		i+1, source.numchunks,
		(offset + len(chunk)) / nsamples * 100.0,
		(offset+len(chunk)) / freq,
		(offset+len(chunk)) / 1e6
	))
	window = signal[offset:offset+len(chunk)]

	for k in source.probes:
		label = source.probes[k]

		window[:,k] = chunk
		window[:,k] >>= k
		window[:,k] &= 1

print
#del source

print "SPI commands:",

cs_delta = np.diff(signal[:,ch_cs])
cs_edges, = cs_delta.nonzero()

cs_active = np.diff(cs_delta[cs_edges])

cs_active_indices = (cs_active > 0).nonzero()[0]

cs_starts = cs_edges[cs_active_indices]
cs_stops = cs_edges[cs_active_indices + 1]

numcommands = len(cs_starts)
print numcommands

def decode_spi(mosi, miso, sclk):
	sample_at, = (np.diff(sclk) > 0).nonzero()

	mosi_bits = mosi[sample_at + 1]
	miso_bits = miso[sample_at + 1]

	return mosi_bits, miso_bits

def round_down(n, k):
	return (n // k) * k

def big_endian(bytes):
	result = 0
	for byte in bytes:
		result = (result << 8) | byte
	return result

def iterate_spi():
	for u,v in zip(cs_starts, cs_stops):
		chunk = signal[u:v]
		mosi = chunk[:, ch_mosi]
		miso = chunk[:, ch_miso]
		sclk = chunk[:, ch_sclk]

		mosi_bits, miso_bits = decode_spi(mosi, miso, sclk)
		length = round_down(len(mosi_bits), 8)

		mosi_bits = mosi_bits[:length]
		miso_bits = miso_bits[:length]

		mosi_bytes = np.packbits(mosi_bits)
		miso_bytes = np.packbits(miso_bits)

		yield (u / freq, mosi_bytes, miso_bytes)

chip = SPIFlash()

coverage = 0
cmdcount = 0

readcount = 0
writecount = 0
erasecount = 0

bytesread = 0
byteswritten = 0
bytes_erased = 0

emptycmds = 0
lastempty = 0
for (t, mosi, miso) in iterate_spi():
	cmdcount += 1

	if len(mosi) == 0:
		emptycmds += 1
		lastempty = t
		continue

	cmd = mosi[0]
	addr = big_endian(mosi[1:4])
	miso_data = miso[4:]
	mosi_data = mosi[4:]

	sys.stdout.write("\r{:.1f}% @ {:.6f}s".format(100 * cmdcount / numcommands, t))

	if cmd == 0x02: # page program
		# 24 bits address
		# no dummy cycles
		# 2048 bits data
		assert len(mosi_data) == 2048 // 8
		writecount += 1
		byteswritten += len(mosi_data)
		chip.write(addr, mosi_data)

	elif cmd == 0xAD: # continuous program
		raise NotImplemented
		
	elif cmd == 0x03: # read
		readcount += 1
		bytesread += len(miso_data)
		chip.write(addr, miso_data)

	elif cmd == 0x0B: # fast read
		readcount += 1
		miso_data = miso_data[1:] # 8 dummy cycles
		bytesread += len(miso_data)
		chip.write(addr, miso_data)

	elif cmd == 0x20: # sector erase
		blocksize = 4 << 10
		chip.erase(addr, blocksize)
		erasecount += 1
		bytes_erased += blocksize

	elif cmd in (0xD8, 0x52): # block erase
		blocksize = {
			0xD8: 64 << 10,
			0x52: 32 << 10,
		}[cmd]
		chip.erase(addr, blocksize)
		erasecount += 1
		bytes_erased += blocksize

	elif cmd in (0x60, 0xc7):
		chip.erase(0, len(chip))



print
if emptycmds: print "{} empty commands (last at {:.6f}s)".format(emptycmds, lastempty)
print "{} read commands, total {} bytes".format(readcount, bytesread)
print "{} write commands, total {} bytes".format(writecount, byteswritten)
print "{} erase commands, total {} bytes".format(erasecount, bytes_erased)
print "{} bytes covered".format(chip.mask.sum())
print "{} bytes of memory assumed".format(len(chip))
print "ranges:"
maskdiff = np.diff(chip.mask)
edges = {
	index+1
	for index in maskdiff.nonzero()[0]
}
if chip.mask[0]: edges.add(0)
for (u,v) in np.uint32(sorted(edges)).reshape((-1, 2)):
	print "{:8x}h - {:8x}h: {:9d} bytes".format(u, v-1, v-u)

chip.save(infile + ".flash")
chip.save_mask(infile + ".mask")
