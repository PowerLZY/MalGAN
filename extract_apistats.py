import glob, os, json

reports_dir = 'your path'#'C:/Users/Kuan/Desktop/malgan_exp*'
dirs = glob.glob(reports_dir)
for d in dirs:
    loadpath = os.path.join(d, 'reports', 'report.json')
    if os.path.exists(loadpath):
        with open(loadpath, 'r') as f:
            data = json.load(f)
            name = data['target']['file']['name']
            print(name)
            if 'behavior' in data:
                if 'apistats' in data['behavior']:
                    data = data['behavior']['apistats']
                    apistats = {}
                    for apistat in data.values():
                        apistats = dict(apistats, **apistat)
                    savejson = {'name': name, 'apistats': apistats, 'class': 'benign'}
                    savepath = os.path.join('../apistats', name[:-3]+'json')
                    with open(savepath, 'w') as s:
                        json.dump(savejson, s, ensure_ascii=False)