import gc
import warnings
from datetime import datetime
from functools import partial
from datasets import Dataset, DatasetDict
from flask import Flask, jsonify, request, make_response
from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer
from Member_matching_utils import *
from XML_to_DF_Member import *
import json
warnings.simplefilter('ignore')

gc.collect()
app = Flask(__name__)

mem_logger = setup_logger('Member')
app.logger.addHandler('Member')

try:
    model_nm = 'D:/Xelp_work/FSL Project/Models/Member_New/'
    tokz = AutoTokenizer.from_pretrained(model_nm)
    model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=2)
    mem_logger.info("Model Loaded")
    trainer = Trainer(model, tokenizer = tokz)
except Exception as exp_model:
    mem_logger.exception(exp_model)

def count_true(row, cols):
    values = row[cols]
    return sum(values)

def compare_pair(row, col1, col2):
    if row[col1] != 'missing':
        return row[col1] == row[col2]
    else:
        return False

def gen_true_match(prepared_data, ocr_cols):
    db_cols = [col[:-1] for col in ocr_cols]
    pairs_to_compare = [(db_col+"_", db_col) for db_col in db_cols]
    new_cols = []
    for col1, col2 in pairs_to_compare:
        new_col_name = f'{col1}_{col2}_match'
        new_cols.append(new_col_name)
        prepared_data[new_col_name] = prepared_data.apply(compare_pair, axis=1, args=(col1, col2))

    prepared_data['total_true'] = prepared_data.apply(count_true, axis=1, args=(new_cols,))
    prepared_data = prepared_data.drop(new_cols, axis=1)
    # prepared_data_with_true = prepared_data[prepared_data['total_true'] > 1]
    # dummy_data = prepared_data[prepared_data['total_true'] <= 1]
    
    # prepared_data['total_true'] = prepared_data['total_true'].astype(str)
    return prepared_data

def re_org_prob(df):
    first_row = df["Prob_Score"][0]
    delta = df["Prob_Score"].max() - df["Prob_Score"][0]
    df["new_score"] = df["Prob_Score"] - delta
    for idx, old_score in enumerate(df["Prob_Score"]):
        if old_score == first_row:
            df.loc[idx, "new_score"] = df.loc[idx, "new_score"] + 2*delta
    df["new_score"] = df["new_score"].apply(lambda x: x if x > 0 else 0)
    df['new_score'] = df['new_score'].apply(lambda x: round(x,4))
    return df

def tok_func(row, tokenizer):
    return tokenizer(row['sequence1'], row['sequence2'])

def apply_tok(test_df, tokz):
    partial_tok_func = partial(tok_func, tokenizer=tokz)
    test_ds = test_df.map(partial_tok_func)
    return DatasetDict({"test":test_ds})

def clean_data(df):
    df_filtered = df.loc[df['Ami'].notna() | df['EmpDepLname'].notna() |  df['EmpDepFname'].notna() | 
                    df['MemberDob'].notna() | df['EmpLastNm'].notna() | df['EmpFirstNm'].notna() |
                    df['EmpDob'].notna() | df['EmpInd'].notna() |df['SubLastName'].notna() | df['SubFirstName'].notna() |
                    df['SubDob'].notna() | df['AmiNum'].notna() |df['DepDob'].notna() | df['DepFirstName'].notna() |
                    df['DepLastName'].notna() | df['SubSsn'].notna()].reset_index(drop=True)
    # print(df_filtered.shape)
    return df_filtered

def move_to_beginning(dictionary, key):
    value = dictionary.pop(key)
    dictionary = {key: value, **dictionary}
    return dictionary


@app.route('/Member_Matching', methods=['GET', 'POST'])
def Member_Matching():
    if request.method == 'POST':
        data = request.get_json(force=True)
        file_name = data['file_name']
        mem_logger.info(file_name)
        prepared_data = member_xml_to_df(file_name)
        prepared_data = clean_data(prepared_data)
        prepared_data = prepared_data.fillna("Missing")
        prepared_data.replace('', "Missing", inplace=True)
        mem_logger.info("Data Frame Created", prepared_data.shape[0])
        
        ocr_cols = ['Ami_', 'EmpDepLname_', 'EmpDepFname_', 'MemberDob_', 'EmpLastNm_', 'EmpFirstNm_', 'EmpDob_', 'EmpInd_', 'SubLastName_', 'SubFirstName_', \
                    'SubDob_', 'AmiNum_', 'DepLastName_','DepFirstName_', 'DepDob_', 'SubSsn_']
        
        db_cols_all = ['Ami', 'EmpDepLname', 'EmpDepFname', 'MemberDob', 'EmpLastNm', 'EmpFirstNm', 'EmpDob', 'EmpAddr1', 'EmpCity', 'EmpState','EmpZip', 'EmpInd', \
                       'EmpZipLast4', 'EmpIdInd', 'EmpAcctNum', 'EmpEffDt', 'EmpCanDt', 'EmpSexCode', 'EmpNumDeps', 'EmpDepId','EmpDepEffDt', 'EmpDepCanDt', 'MemberSex', \
                        'Platform', 'SubLastName', 'SubFirstName', 'SubDob', 'AmiNum', 'DepLastName', 'DepFirstName', 'DepDob', 'Addr1', 'City', 'State', 'Zip', 'SubSsn', 'EeIdInd', \
               'MNum', 'DepNum', 'ClientGrp', 'SubEffDate', 'SubDntlOfc', 'SubGenderCode', 'SubRelCode', 'ClaimOfcInd', 'GrpType', 'DepEffDate', 'DepDntlOfc', 'DepGenderCode', 'dtUpload', 'InsrdsFco']
        
        dup_cols = ["Ami", "EmpDepLname", "EmpDepFname", "MemberDob", "EmpLastNm", "EmpFirstNm", "EmpDob", "EmpAddr1", "EmpCity","EmpState","EmpZip", "SubLastName", \
                    "SubFirstName", "SubDob", "AmiNum", "DepLastName", "DepFirstName", "DepDob","Addr1", "City", "State", "Zip", "SubSsn", "EmpCanDt"] 
        prepared_data = prepared_data[ocr_cols + db_cols_all +["QueryNumber","id"]]
        try:
            if prepared_data.shape[0] > 0:
                prepared_data = prepared_data.apply(lambda x:x.str.lower())
                prepared_data = prepared_data.drop_duplicates(subset=dup_cols)
                prepared_data["Record_id"] = prepared_data["QueryNumber"] +"_"+prepared_data["id"]
                prepared_data = gen_true_match(prepared_data, ocr_cols)
                prepared_data['sequence1'] = prepared_data.apply(create_seq1, axis=1)
                prepared_data['sequence2'] = prepared_data.apply(create_seq2, axis=1)
                prepared_data['sequence1'] = prepared_data['sequence1'].str.replace("missing", " [u] ")
                prepared_data['sequence2'] = prepared_data['sequence2'].str.replace("missing", " [u] ")
                test_ds = Dataset.from_pandas(prepared_data)
                test_ds = apply_tok(test_ds, tokz)
                preds, _, _ = trainer.predict(test_ds['test'])
                del test_ds
                gc.collect()
                probs_score = softmax(preds, axis=-1)
                prepared_data['Prob_Score'] = probs_score[:,1]
                prepared_data['Prob_Score'] = prepared_data['Prob_Score'].apply(lambda x: round(x,4))
                cols_list  = ocr_cols + db_cols_all + ["Prob_Score", "total_true", "sequence1", "sequence2", "Record_id"]
                output_df = prepared_data[cols_list].sort_values(by = ['total_true', 'Prob_Score'], ascending = [False, False], key=None)
                output_df = output_df.reset_index(drop=True)
                mem_logger.info("Output Data Frame Size::  %s", output_df.shape)
                result = dict(zip(output_df["Record_id"], output_df["Prob_Score"]))
                
                STP_dict = {}

                is_emp_id_match = False
                if output_df.iloc[0]["Ami_"] != 'missing':
                    if output_df.iloc[0]["Ami_"] == output_df.iloc[0]["Ami"]:
                        is_emp_id_match = True
                elif output_df.iloc[0]["EmpInd_"] != 'missing':
                    if output_df.iloc[0]["EmpInd_"] == output_df.iloc[0]["EmpInd"]:
                        is_emp_id_match = True
                else:
                    is_emp_id_match = False
                
               
                if (is_emp_id_match) and \
                    (output_df.iloc[0]["MemberDob_"] == output_df.iloc[0]["MemberDob"]) and (output_df.iloc[0]["EmpLastNm_"] == output_df.iloc[0]["EmpLastNm"]) and \
                    (output_df.iloc[0]["EmpDob_"] == output_df.iloc[0]["EmpDob"]) and  (output_df.iloc[0]['EmpDepLname_'] == output_df.iloc[0]['EmpDepLname']) and \
                    (output_df.iloc[0]['EmpDepFname_'] == output_df.iloc[0]['EmpDepFname']) and (output_df.iloc[0]['SubLastName_'] == output_df.iloc[0]['SubLastName']) and \
                    (output_df.iloc[0]['SubFirstName_'] == output_df.iloc[0]['SubFirstName']) and \
                    (output_df.iloc[0]["EmpFirstNm_"] == output_df.iloc[0]["EmpFirstNm"]) and (output_df.iloc[0]["total_true"] >= 6) and (output_df.iloc[0]["Prob_Score"] >= 0.99) :
                    STP_dict["STP"] = output_df.iloc[0]["Record_id"]               
                else:
                    STP_dict["STP"] = False
                
                if (output_df.iloc[0]['Ami_'] == 'missing') and (output_df.iloc[0]['EmpDepLname_'] == 'missing') and (output_df.iloc[0]['EmpDepFname_']== 'missing') and \
                    (output_df.iloc[0]['MemberDob_'] == 'missing') and (output_df.iloc[0]['EmpLastNm_'] == 'missing') and (output_df.iloc[0]['EmpFirstNm_'] == 'missing') and \
                    (output_df.iloc[0]['EmpDob_'] == 'missing') and (output_df.iloc[0]['EmpInd_'] == 'missing'):
                        if (output_df.iloc[0]['SubLastName_'] == output_df.iloc[0]['SubLastName']) and (output_df.iloc[0]['SubFirstName_'] == output_df.iloc[0]['SubFirstName']) and \
                            (output_df.iloc[0]['SubDob_'] == output_df.iloc[0]['SubDob']) and (output_df.iloc[0]['AmiNum_'] == output_df.iloc[0]['AmiNum']) and \
                            (output_df.iloc[0]['DepLastName_'] == output_df.iloc[0]['DepLastName']) and (output_df.iloc[0]['DepFirstName_'] == output_df.iloc[0]['DepFirstName']):
                            
                            STP_dict["STP"] = output_df.iloc[0]["Record_id"]
                        else:
                            STP_dict["STP"] = False
                else:
                    STP_dict["STP"] = False

                mem_logger.info("STP::   %s", str(STP_dict["STP"]))
                final_dict = {"STP":STP_dict["STP"], "Prob_scores":result}
                return json.dumps(final_dict)
            else:
                mem_logger.error("All 16 OCR/DB Fields missing in XML File")
                return json.dumps({"Error":"All 16 OCR/DB Fields missing in XML File"})
        
        except Exception as exp:
            mem_logger.exception(exp)
            return json.dumps({"Exception":"Exception in Parsing the XML File"})

if __name__ == '__main__':
    app.run(host='localhost',port='5002',debug=True)
