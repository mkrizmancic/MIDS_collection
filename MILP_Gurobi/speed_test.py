import math
import networkx as nx
import matplotlib.pyplot as plt

from milpMIDS import find_MIDS


def main():
	max_time = 1
	step = 5
	repeat = 5

	n = 0
	sol = True
	elps = 0
	avg = 0
	x = []
	y = []

	while sol and elps <= max_time:
		x.append(n)
		y.append(elps)

		n += step

		print(f"Searching MIDS for {n=}...", end='')
		for _ in range(repeat):
			G = nx.connected_watts_strogatz_graph(n, max(int(math.sqrt(n)), 2), 0.5)
			sol, elps = find_MIDS(G, 'MIDS', outputFlag=0)
			if sol:
				avg += elps
			else:
				print('FAILED')
				break
		else:
			elps = avg / repeat
			print('SUCCESS')


	fig = plt.figure()
	ax = fig.add_subplot(1, 2, 1)
	plt.stem(x, y)
	
	ax = fig.add_subplot(1, 2, 2)
	plt.stem(x, y)
	ax.set_yscale('log')

	plt.show()


if __name__ == '__main__':
	main()