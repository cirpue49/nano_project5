import xml.etree.cElementTree as ET
osm_file = open(san_francisco_california.osm, "r")

count=0
for event, elem in ET.iterparse(osm_file, events=("start",)):
    print elem
    count+=1
    if count ==50:
        break