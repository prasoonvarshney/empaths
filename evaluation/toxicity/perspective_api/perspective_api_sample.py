from googleapiclient import discovery
import json

API_KEY = 'AIzaSyAOf7k6AwhH3MRxCiWJNeHUEe49_Lck-J0'

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

analyze_request_rude = {
  'comment': { 'text': 'Why would anyone promote you?' },
  'requestedAttributes': {'TOXICITY': {}, "languages": ["en"]}
}

analyze_request_emp = {
  'comment': { 'text': 'Congrats! That’s great!' },
  'requestedAttributes': {'TOXICITY': {}}
}

response = client.comments().analyze(body=analyze_request_rude).execute()
print(json.dumps(response, indent=2))

response = client.comments().analyze(body=analyze_request_emp).execute()
print(json.dumps(response, indent=2))