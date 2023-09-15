import pandas as pd
import xml.etree.ElementTree as ET
from Member_matching_utils import *

def member_xml_to_df(filepath):
    col_list = ['Ami', 'EmpDepLname', 'EmpDepFname', 'MemberDob', 'EmpLastNm', 'EmpFirstNm', 'EmpDob', 'EmpAddr1', 'EmpAddr2', 'EmpCity', 'EmpState', 'EmpZip', 'EmpInd', 'EmpZipLast4', 'EmpIdInd', 'EmpAcctNum', 'EmpEffDt', 'EmpCanDt', 'EmpSexCode', 'EmpNumDeps', 'EmpDepId', 'EmpDepEffDt', 'EmpDepCanDt', 'MemberSex', 'Platform', 'SubLastName', 'SubFirstName', 'SubDob', 'AmiNum', 'DepLastName', 'DepFirstName', 'DepDob', 'Addr1', 'Addr2', 'City', 'State', 'Zip', 'SubSsn', 'EeIdInd', 'MNum', 'DepNum', 'ClientGrp', 'SubEffDate', 'SubInactiveDate', 'SubDntlOfc', 'SubGenderCode', 'SubRelCode', 'ClaimOfcInd', 'GrpType', 'DepId', 'DepEffDate', 'DepInactiveDate', 'DepDntlOfc', 'DepGenderCode', 'DepRelCode', 'dtUpload','Action', 'InsrdsFco', 'SubSsn1', 'SubSsn2', 'SubSsn3', 'QueryNumber', 'id' , 'Selected_by_agent', 'Folder', 'FileName']
    l0 = l = []

    [l0.append((col_list[i]+'_')) for i in range(len(col_list)-8)]
    l = l0 + col_list

    main_cols = []
    [main_cols.append(i) for i in l if i not in main_cols]

    tree=ET.parse(filepath)
    root = tree.getroot()
    querylist_tag = root.find('LookUps/Lkup/QueryList')
    feeds_tag = root.findall('LookUps/Lkup/Feeds')
    result_tag = root.find('LookUps/Lkup/Result/Record')

    if len(querylist_tag) != len(feeds_tag):
        return;

    df = pd.DataFrame(columns = main_cols)

    row = 0
    for i in range(len(querylist_tag)):
        feeds = feeds_tag[i]

        qid = feeds.attrib['Qid']
        cols = feeds.attrib['Cols']
        extracted_col_names = cols.split('»')

        records = feeds.findall('Record')
        if len(records) == 0:
            for j in querylist_tag[i]:
                attribute = j.attrib['Name']
                df.loc[row, (attribute+'_')] = j.text
            df.loc[row, 'QueryNumber'] = qid
            df.loc[row, 'FileName'] = filepath.split("/")[-1]
            row += 1

        for record in records:
            for j in querylist_tag[i]:
                attribute = j.attrib['Name']
                df.loc[row, (attribute+'_')] = j.text
            df.loc[row, 'QueryNumber'] = qid
            df.loc[row, 'id'] = record.attrib['Id']
            df.loc[row, 'FileName'] = filepath.split("/")[-1]
            Text = record.text
            col_data = Text.split('»')

            for k in range(len(extracted_col_names)):
                df.loc[row, extracted_col_names[k]] = col_data[k]
            row += 1
    try:
        result_id = result_tag.attrib['Id']
        result_query = result_tag.attrib['Qid']
        for i in range(len(df)):
            if df['QueryNumber'][i] == result_query and df['id'][i] == result_id:
                df.loc[i, 'Selected_by_agent'] = 1
            else:
                df.loc[i, 'Selected_by_agent'] = 0
    except:
        pass
    
    return df