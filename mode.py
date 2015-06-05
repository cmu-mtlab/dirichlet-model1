import sys
from collections import defaultdict

def argmax(d):
	best = None
	for k, v in d.iteritems():	
		if best == None or v > best[1]:	
			best = (k, v)
	return best

sentence_count = int(sys.argv[1])
burn_in = int(sys.argv[2]) if len(sys.argv) >= 3 else 0

iteration = 0
sentence_index = 0
line_num = 1

link_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

sys.stderr.write('Taking the mode over %d sentences with a burn-in of %d iterations\n' % (sentence_count, burn_in))
sys.stderr.write('Reading iteration %d\r' % 0)
for line in sys.stdin:
	if iteration >= burn_in:
		n = (line_num - 1) % (sentence_count)
		if n < 150:
			line = line.strip()
			links = line.split(' ')
			links = [link.split('-') for link in links if len(link) > 0]
			links = [(int(link[0]), int(link[1])) for link in links]
			#links = [(int(link[0]) - 1, int(link[1])) for link in links if link[0] != '0']
			for (i, j) in links:
				link_counts[n][j][i] += 1

	if line_num % (sentence_count) == 0:
		sentence_index = 0
		iteration += 1
		sys.stderr.write('Reading iteration %d\r' % iteration)

	line_num += 1
sys.stderr.write('\n')

for n in range(sentence_count):
	for j in sorted(link_counts[n].keys()):
		i, v = argmax(link_counts[n][j])
		s = sum(link_counts[n][j].values())
		if v >= (iteration - burn_in) - s:
			print '%d-%d' % (i,j),
	print
