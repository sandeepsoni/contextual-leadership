def read_cascades_from_file (cascades_file, sep="\t"):
	cascades = dict ()
	with open (cascades_file) as fin:
		for line in fin:
			parts = line.strip().split (sep)
			innov = parts[0]
			year = int (parts[1])
			paper_id = parts[2]

			if innov not in cascades:
				cascades[innov] = list ()
			cascades[innov].append ((paper_id, year))
			
	serialized = [cascades[innov] for innov in cascades]
	innovs = [innov for innov in cascades]
	return serialized, innovs
