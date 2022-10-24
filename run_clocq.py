# Usage: python run_clocq.py <PATH_TO_INPUT> <PATH_TO_OUTPUT> [<PATH_TO_CONFIG>]
import json

from tqdm import tqdm
from clocq.interface.CLOCQInterfaceClient import CLOCQInterfaceClient

from main import get_config

def prune_predicates(linkings):
	"""Function to prune predicates from results."""
	linkings = list(set(linkings))
	linkings = [linking for linking in linkings if linking and linking[0] == "Q"]
	return linkings

if __name__ == "__main__":
	if len(sys.argv) < 2:
		raise Exception(
			"Usage: python run_clocq.py <PATH_TO_INPUT> <PATH_TO_OUTPUT> [<PATH_TO_CONFIG>]"
		)
	# load params
	input_path = sys.argv[1]
	output_path = sys.argv[2]
	config_path = sys.argv[3] if len(sys.argv) > 2 else "config.yml"
	config = get_config(config_path)

	# initiate CLOCQ
	clocq = CLOCQInterfaceClient(host=config["clocq_url"], port=str(config["clocq_port"]))
	clocq_params = {
		"k": config["clocq_k"],
		"p": config["clocq_p"]
	}

	# open data
	with open(input_path, "r") as fp:
		data = json.load(fp)

	# process data
	for instance in tqdm(data):
		question = instance["question"]
		res = clocq.get_search_space(question, parameters=clocq_params)
		linkings = [kb_item["item"]["id"] for kb_item in res["kb_item_tuple"]]
		instance["kb_item_tuple"] = res["kb_item_tuple"]
		linkings = prune_predicates(linkings)
		# keep gold entities during training
		if "entities" in instance:
			instance["gold_entities"] = instance["entities"]
		instance["entities"] = linkings

	# store results
	with open(output_path, "w") as fp:
		fp.write(json.dumps(data, indent=4))