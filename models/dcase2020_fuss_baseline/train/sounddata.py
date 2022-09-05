personid=['431', '20n', '20m', '407', '444', '01w', '20b', '20c', '012', '018', '020', '420', '40m', '442', '432', '02c', '207', '011', '01x', '20h', '20t', '447', '443', '026', '028', '002', '40j', '01u', '20s', '01p', '446', '02d', '01e', '022', '40h', '01i', '20o', '20p', '20u', '001', '01y', '019', '00c', '01n', '404', '20g', '423', '20i', '01l', '016', '405', '01k', '20v', '40f', '20f', '406', '01j', '023', '20j', '00b', '015', '40o', '013', '01o', '22g', '02a', '01v', '20k', '024', '40g', '421', '20r', '403', '01b', '051', '01t', '40d', '02b', '400', '01d', '205', '01a', '052', '440', '40p', '203', '209', '01q', '40a', '40i', '20l', '409', '422', '441', '00d', '204', '40l', '40e', '22h', '01s', '20q', '025', '00f', '208', '01c', '40c', '445', '20a', '021', '01f', '050', '01r', '029', '20e', '02e', '20d', '017', '40k', '00a', '027', '053', '01m', '01z', '206', '430', '40b', '408', '01g', '014', '40n', '401']
noiseclass = ["bird","counstructionSite","crowd","foutain","park","rain","schoolyard","traffic","ventilation","wind_tree"]
needpersion=["400", "002", "001","431","430","432"]
zeros=['zeros']
soundclass=[]
#soundclass.extend(personid)
soundclass.extend(needpersion)
soundclass.extend(noiseclass)
soundclass.extend(zeros)
soundclass = list(set(soundclass))#to unique
print(soundclass)#ok
classnum = len(soundclass)
classdict = {k: v for v, k in enumerate(soundclass)}