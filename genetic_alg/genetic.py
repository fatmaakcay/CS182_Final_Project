import random


def mutate(chromosome, n):
	for x in xrange(n):
		p = random.randint(0, len(chromosome)-1)
		if random.random > 0.5:
			chromosome[p] += random.random() * 0.01
		else:
			chromosome[p] -= random.random() * 0.01
	return chromosome

def recombine(chr1, chr2):
	child = []

	for i in range(len(chr1)):
		if random.random() > 0.5:
			child.append(chr1[i])
		else:
			child.append(chr2[i])

	return child



