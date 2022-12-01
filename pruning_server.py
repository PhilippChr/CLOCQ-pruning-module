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
@app.route("/entity_linking", methods=["GET", "POST"])
def entity_linking():
	if request.json:
		params_dict = request.json
	else:
		params_dict = request.args
	# load question
	question = params_dict.get("question")
	if question is None:
		return jsonify(None)
	# load parameters: possibility to give specific parameters
	parameters = params_dict.get("parameters")
	if parameters is None:
		parameters = copy.deepcopy(DEF_PARAMS)
	else:
		new_parameters = copy.deepcopy(DEF_PARAMS)
		for key in parameters:
			new_parameters[key] = parameters[key]
		parameters = new_parameters
	# load k: num. of relations per mention -> prefer explicitly given k
	if "k" in params_dict:
		k = params_dict.get("k")
		parameters["k"] = params_dict.get("k")
	else:
		k = "AUTO"
	parameters["p_setting"] = 1

	# run CLOCQ
	res = clocq.get_search_space(question, parameters=parameters)
	linkings = res["kb_item_tuple"]

	# prune irrelevant linkings
	linkings, mentions = pruning_module.get_entity_linkings(question, linkings, k)
	res = {
		"linkings": linkings,
		"mentions": mentions
	}
	return jsonify(res)


@app.route("/relation_linking", methods=["GET", "POST"])
def relation_linking():
	if request.json:
		params_dict = request.json
	else:
		params_dict = request.args
	# load question
	question = params_dict.get("question")
	if question is None:
		return jsonify(None)
	# load k: num. of relations per mention
	top_ranked = params_dict.get("top_ranked")
	if top_ranked is not None and (top_ranked == False or top_ranked == "0" or top_ranked == "False"):
		top_ranked = False
	else:
		top_ranked = True
	# load parameters: possibility to give specific parameters
	parameters = params_dict.get("parameters")
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
