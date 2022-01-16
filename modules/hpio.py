def read_cascades_from_file (cascades_file, sep="\t"):
	idx, iidx = dict (), dict ()
	cascades = dict ()
	with open (cascades_file) as fin:
		for line in fin:
			parts = line.strip().split (sep)
			innov = parts[0]
			year = int (parts[1])
			paper_id = parts[2]

			if paper_id not in idx:
				idx[paper_id] = len (idx)
				iidx[idx[paper_id]] = paper_id
			
			if innov not in cascades:
				cascades[innov] = list ()
			cascades[innov].append ((idx[paper_id], year))
			
	serialized = [cascades[innov] for innov in cascades]
	innovs = [innov for innov in cascades]
	return idx, iidx, serialized, innovs
