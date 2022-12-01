import os
import json
import copy
import datetime

from flask import Flask, jsonify, render_template, request, session

from pruning_module import PruningModule
from clocq.interface.CLOCQInterfaceClient import CLOCQInterfaceClient
from clocq.config import DEF_PARAMS

"""Flask config"""
app = Flask(__name__)
# Set the secret key to some random bytes.
app.secret_key = os.urandom(32)
app.permanent_session_lifetime = datetime.timedelta(days=365)

"""Instantiate modules"""
pruning_module = PruningModule()
config = pruning_module.config
clocq = CLOCQInterfaceClient(config["clocq_url"], config["clocq_port"])


"""API endpoints"""
@app.route("/entity_linking", methods=["POST"])
def entity_linking():
	json_dict = request.json
	# load question
	question = json_dict.get("question")
	if question is None:
		return None
	# load parameters: possibility to give specific parameters
	parameters = json_dict.get("parameters")
	if parameters is None:
		parameters = copy.deepcopy(DEF_PARAMS)
	else:
		new_parameters = copy.deepcopy(DEF_PARAMS)
		for key in parameters:
			new_parameters[key] = parameters[key]
		parameters = new_parameters
	# load k: num. of relations per mention -> prefer explicitly given k
	if "k" in json_dict:
		parameters["k"] = json_dict.get("k")
	parameters["p_setting"] = 1

	# run CLOCQ
	res = clocq.get_search_space(question, parameters=parameters)
	linkings = res["kb_item_tuple"]

	# prune irrelevant linkings
	linkings, mentions = pruning_module.get_entity_linkings(question, linkings)
	res = {
		"linkings": linkings,
		"mentions": mentions
	}
	return jsonify(res)


@app.route("/relation_linking", methods=["POST"])
def relation_linking():
	json_dict = request.json
	# load question
	question = json_dict.get("question")
	if question is None:
		return None
	# load k: num. of relations per mention
	top_ranked = json_dict.get("top_ranked")
	if top_ranked is None or top_ranked:
		top_ranked = True
	else:
		top_ranked = False
	# load parameters: possibility to give specific parameters
	parameters = json_dict.get("parameters")
	if parameters is None:
		parameters = copy.deepcopy(DEF_PARAMS)
	else:
		new_parameters = copy.deepcopy(DEF_PARAMS)
		for key in parameters:
			new_parameters[key] = parameters[key]
		parameters = new_parameters
	parameters = copy.deepcopy(DEF_PARAMS)
	# set parameters dedicated for RL 
	parameters["k"] = 40
	parameters["d"] = 50
	parameters["p_setting"] = 1
	
	# run CLOCQ
	res = clocq.get_search_space(question, parameters=parameters)
	linkings = res["kb_item_tuple"]

	# prune irrelevant linkings
	linkings, mentions = pruning_module.get_relation_linkings(question, linkings, top_ranked)
	res = {
		"linkings": linkings,
		"mentions": mentions
	}
	return jsonify(res)


if __name__ == "__main__":
	app.run(host=config["host"], port=config["port"], threaded=True)
