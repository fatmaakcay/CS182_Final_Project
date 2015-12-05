import random


def mutate(chromosome, n):
	for x in xrange(n):
		p = random.randint(0, len(chromosome)-1)
		chromosome[p] = random.random() % 0.15
	return chromosome

def recombine(chr1, chr2):
	child = []

	r = random.randint(0,1)

	# recombine in 1 place
	if r == 0:
		p = random.randint(0, len(chr1))
		child.extend(chr1[:p])
		child.extend(chr2[p:])
	else: #recombine in 2 places 
		p = random.randint(0, len(chr1)/2)
		q = random.randint((len(chr1)/2)+1, len(chr1))
		child.extend(chr1[:p])
		child.extend(chr2[p:q])
		child.extend(chr2[q:])

	return child



