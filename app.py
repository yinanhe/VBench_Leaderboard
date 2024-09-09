__all__ = ['block', 'make_clickable_model', 'make_clickable_user', 'get_submissions']
import os
import io
import gradio as gr
import pandas as pd
import json
import shutil
import tempfile
import datetime
import zipfile


from constants import *
from huggingface_hub import Repository
HF_TOKEN = os.environ.get("HF_TOKEN")

global data_component, filter_component


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths

def add_new_eval(
    input_file,
    model_name_textbox: str,
    revision_name_textbox: str,
    model_link: str,
    team_name: str,
    contact_email: str
):
    if input_file is None:
        return "Error! Empty file!"
    if  model_link == '' or model_name_textbox == '' or contact_email == '':
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
    # upload_data=json.loads(input_file)
    upload_content = input_file
    submission_repo = Repository(local_dir=SUBMISSION_NAME, clone_from=SUBMISSION_URL, use_auth_token=HF_TOKEN, repo_type="dataset")
    submission_repo.git_pull()
    filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    now = datetime.datetime.now()
    with open(f'{SUBMISSION_NAME}/{filename}.zip','wb') as f:
        f.write(input_file)
    # shutil.copyfile(CSV_DIR, os.path.join(SUBMISSION_NAME, f"{input_file}"))

    csv_data = pd.read_csv(CSV_DIR)

    if revision_name_textbox == '':
        col = csv_data.shape[0]
        model_name = model_name_textbox
    else:
        model_name = revision_name_textbox
        model_name_list = csv_data['Model Name (clickable)']
        name_list = [name.split(']')[0][1:] for name in model_name_list]
        if revision_name_textbox not in name_list:
            col = csv_data.shape[0]
        else:
            col = name_list.index(revision_name_textbox)    
    if model_link == '':
        model_name = model_name  # no url
    else:
        model_name = '[' + model_name + '](' + model_link + ')'

    os.makedirs(filename, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(input_file), 'r') as zip_ref:
        zip_ref.extractall(filename)

    upload_data = {}
    for file in os.listdir(filename):
        if file.startswith('.') or file.startswith('__'):
            print(f"Skip the file: {file}")
            continue
        cur_file = os.path.join(filename, file)
        if os.path.isdir(cur_file):
            for subfile in os.listdir(cur_file):
                if subfile.endswith(".json"):
                    with open(os.path.join(cur_file, subfile)) as ff:
                        cur_json = json.load(ff)
                        print(file, type(cur_json))
                        if isinstance(cur_json, dict):
                            print(cur_json.keys())
                            for key in cur_json:
                                upload_data[key.replace('_',' ')] = cur_json[key][0]
                                print(f"{key}:{cur_json[key][0]}")
        elif cur_file.endswith('json'):
            with open(cur_file) as ff:
                cur_json = json.load(ff)
                print(file, type(cur_json))
                if isinstance(cur_json, dict):
                    print(cur_json.keys())
                    for key in cur_json:
                        upload_data[key.replace('_',' ')] = cur_json[key][0]
                        print(f"{key}:{cur_json[key][0]}")
    # add new data
    new_data = [model_name]
    print('upload_data:', upload_data)
    for key in TASK_INFO:
        if key in upload_data:
            new_data.append(upload_data[key])
        else:
            new_data.append(0)
    if team_name =='' or 'vbench' in team_name.lower():
        new_data.append("User Upload")
    else:
        new_data.append(team_name)
    new_data.append(contact_email)
    csv_data.loc[col] = new_data
    csv_data = csv_data.to_csv(CSV_DIR, index=False)
    submission_repo.push_to_hub()
    print("success update", model_name)
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

def get_normalized_df(df):
    # final_score = df.drop('name', axis=1).sum(axis=1)
    # df.insert(1, 'Overall Score', final_score)
    normalize_df = df.copy().fillna(0.0)
    for column in normalize_df.columns[1:-2]:
        min_val = NORMALIZE_DIC[column]['Min']
        max_val = NORMALIZE_DIC[column]['Max']
        normalize_df[column] = (normalize_df[column] - min_val) / (max_val - min_val)
    return normalize_df

def get_normalized_i2v_df(df):
    normalize_df = df.copy().fillna(0.0)
    for column in normalize_df.columns[1:]:
        min_val = NORMALIZE_DIC_I2V[column]['Min']
        max_val = NORMALIZE_DIC_I2V[column]['Max']
        normalize_df[column] = (normalize_df[column] - min_val) / (max_val - min_val)
    return normalize_df


def calculate_selected_score(df, selected_columns):
    # selected_score = df[selected_columns].sum(axis=1)
    selected_QUALITY = [i for i in selected_columns if i in QUALITY_LIST]
    selected_SEMANTIC = [i for i in selected_columns if i in SEMANTIC_LIST]
    selected_quality_score = df[selected_QUALITY].sum(axis=1)/sum([DIM_WEIGHT[i] for i in selected_QUALITY])
    selected_semantic_score = df[selected_SEMANTIC].sum(axis=1)/sum([DIM_WEIGHT[i] for i in selected_SEMANTIC ])
    if selected_quality_score.isna().any().any() and selected_semantic_score.isna().any().any():
        selected_score =  (selected_quality_score * QUALITY_WEIGHT + selected_semantic_score * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)
        return selected_score.fillna(0.0)
    if selected_quality_score.isna().any().any():
        return selected_semantic_score
    if selected_semantic_score.isna().any().any():
        return selected_quality_score
    # print(selected_semantic_score,selected_quality_score )
    selected_score =  (selected_quality_score * QUALITY_WEIGHT + selected_semantic_score * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)
    return selected_score.fillna(0.0)

def calculate_selected_score_i2v(df, selected_columns):
    # selected_score = df[selected_columns].sum(axis=1)
    selected_QUALITY = [i for i in selected_columns if i in I2V_QUALITY_LIST]
    selected_I2V = [i for i in selected_columns if i in I2V_LIST]
    selected_quality_score = df[selected_QUALITY].sum(axis=1)/sum([DIM_WEIGHT_I2V[i] for i in selected_QUALITY])
    selected_i2v_score = df[selected_I2V].sum(axis=1)/sum([DIM_WEIGHT_I2V[i] for i in selected_I2V ])
    if selected_quality_score.isna().any().any() and selected_i2v_score.isna().any().any():
        selected_score =  (selected_quality_score * I2V_QUALITY_WEIGHT + selected_i2v_score * I2V_WEIGHT) / (I2V_QUALITY_WEIGHT + I2V_WEIGHT)
        return selected_score.fillna(0.0)
    if selected_quality_score.isna().any().any():
        return selected_i2v_score
    if selected_i2v_score.isna().any().any():
        return selected_quality_score
    # print(selected_i2v_score,selected_quality_score )
    selected_score =  (selected_quality_score * I2V_QUALITY_WEIGHT + selected_i2v_score * I2V_WEIGHT) / (I2V_QUALITY_WEIGHT + I2V_WEIGHT)
    return selected_score.fillna(0.0)

def get_final_score(df, selected_columns):
    normalize_df = get_normalized_df(df)
    #final_score = normalize_df.drop('name', axis=1).sum(axis=1)
    for name in normalize_df.drop('Model Name (clickable)', axis=1).drop('Source', axis=1).drop('Mail', axis=1):
        normalize_df[name] = normalize_df[name]*DIM_WEIGHT[name]
    quality_score = normalize_df[QUALITY_LIST].sum(axis=1)/sum([DIM_WEIGHT[i] for i in QUALITY_LIST])
    semantic_score = normalize_df[SEMANTIC_LIST].sum(axis=1)/sum([DIM_WEIGHT[i] for i in SEMANTIC_LIST ])
    final_score =  (quality_score * QUALITY_WEIGHT + semantic_score * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)
    if 'Total Score' in df:
        df['Total Score'] = final_score
    else:
        df.insert(1, 'Total Score', final_score)
    if 'Semantic Score' in df:
        df['Semantic Score'] = semantic_score
    else:
        df.insert(2, 'Semantic Score', semantic_score)
    if 'Quality Score' in df:
        df['Quality Score'] = quality_score
    else:
        df.insert(3, 'Quality Score', quality_score)
    selected_score = calculate_selected_score(normalize_df, selected_columns)
    if 'Selected Score' in df:
        df['Selected Score'] = selected_score
    else:
        df.insert(1, 'Selected Score', selected_score)
    return df

def get_final_score_i2v(df, selected_columns):
    normalize_df = get_normalized_i2v_df(df)
    #final_score = normalize_df.drop('name', axis=1).sum(axis=1)
    for name in normalize_df.drop('Model Name (clickable)', axis=1).drop('Video-Text Camera Motion', axis=1):
        normalize_df[name] = normalize_df[name]*DIM_WEIGHT_I2V[name]
    quality_score = normalize_df[I2V_QUALITY_LIST].sum(axis=1)/sum([DIM_WEIGHT_I2V[i] for i in I2V_QUALITY_LIST])
    i2v_score = normalize_df[I2V_LIST].sum(axis=1)/sum([DIM_WEIGHT_I2V[i] for i in I2V_LIST ])
    final_score =  (quality_score * I2V_QUALITY_WEIGHT + i2v_score * I2V_WEIGHT) / (I2V_QUALITY_WEIGHT + I2V_WEIGHT)
    if 'Total Score' in df:
        df['Total Score'] = final_score
    else:
        df.insert(1, 'Total Score', final_score)
    if 'I2V Score' in df:
        df['I2V Score'] = i2v_score
    else:
        df.insert(2, 'I2V Score', i2v_score)
    if 'Quality Score' in df:
        df['Quality Score'] = quality_score
    else:
        df.insert(3, 'Quality Score', quality_score)
    selected_score = calculate_selected_score_i2v(normalize_df, selected_columns)
    if 'Selected Score' in df:
        df['Selected Score'] = selected_score
    else:
        df.insert(1, 'Selected Score', selected_score)
    return df



def get_final_score_quality(df, selected_columns):
    normalize_df = get_normalized_df(df)
    for name in normalize_df.drop('Model Name (clickable)', axis=1):
        normalize_df[name] = normalize_df[name]*DIM_WEIGHT[name]
    quality_score = normalize_df[QUALITY_TAB].sum(axis=1) / sum([DIM_WEIGHT[i] for i in QUALITY_TAB])

    if 'Quality Score' in df:
        df['Quality Score'] = quality_score
    else:
        df.insert(1, 'Quality Score', quality_score)
    # selected_score = normalize_df[selected_columns].sum(axis=1) / len(selected_columns)
    selected_score = normalize_df[selected_columns].sum(axis=1)/sum([DIM_WEIGHT[i] for i in selected_columns])
    if 'Selected Score' in df:
        df['Selected Score'] = selected_score
    else:
        df.insert(1, 'Selected Score', selected_score)
    return df



def get_baseline_df():
    submission_repo = Repository(local_dir=SUBMISSION_NAME, clone_from=SUBMISSION_URL, use_auth_token=HF_TOKEN, repo_type="dataset")
    submission_repo.git_pull()
    df = pd.read_csv(CSV_DIR)
    df = get_final_score(df, checkbox_group.value)
    df = df.sort_values(by="Selected Score", ascending=False)
    present_columns = MODEL_INFO + checkbox_group.value
    df = df[present_columns]
    df = convert_scores_to_percentage(df)
    return df

def get_baseline_df_quality():
    submission_repo = Repository(local_dir=SUBMISSION_NAME, clone_from=SUBMISSION_URL, use_auth_token=HF_TOKEN, repo_type="dataset")
    submission_repo.git_pull()
    df = pd.read_csv(QUALITY_DIR)
    df = get_final_score_quality(df, checkbox_group_quality.value)
    df = df.sort_values(by="Selected Score", ascending=False)
    present_columns = MODEL_INFO_TAB_QUALITY + checkbox_group_quality.value
    df = df[present_columns]
    df = convert_scores_to_percentage(df)
    return df

def get_baseline_df_i2v():
    submission_repo = Repository(local_dir=SUBMISSION_NAME, clone_from=SUBMISSION_URL, use_auth_token=HF_TOKEN, repo_type="dataset")
    submission_repo.git_pull()
    df = pd.read_csv(I2V_DIR)
    df = get_final_score_i2v(df, checkbox_group_i2v.value)
    df = df.sort_values(by="Selected Score", ascending=False)
    present_columns = MODEL_INFO_TAB_I2V + checkbox_group_i2v.value
    df = df[present_columns]
    df = convert_scores_to_percentage(df)
    return df

def get_baseline_df_long():
    submission_repo = Repository(local_dir=SUBMISSION_NAME, clone_from=SUBMISSION_URL, use_auth_token=HF_TOKEN, repo_type="dataset")
    submission_repo.git_pull()
    df = pd.read_csv(LONG_DIR)
    df = get_final_score(df, checkbox_group.value)
    df = df.sort_values(by="Selected Score", ascending=False)
    present_columns = MODEL_INFO + checkbox_group.value
    df = df[present_columns]
    df = convert_scores_to_percentage(df)
    return df

def get_all_df(selected_columns, dir=CSV_DIR):
    submission_repo = Repository(local_dir=SUBMISSION_NAME, clone_from=SUBMISSION_URL, use_auth_token=HF_TOKEN, repo_type="dataset")
    submission_repo.git_pull()
    df = pd.read_csv(dir)
    df = get_final_score(df, selected_columns)
    df = df.sort_values(by="Selected Score", ascending=False)
    return df
    
def get_all_df_quality(selected_columns, dir=QUALITY_DIR):
    submission_repo = Repository(local_dir=SUBMISSION_NAME, clone_from=SUBMISSION_URL, use_auth_token=HF_TOKEN, repo_type="dataset")
    submission_repo.git_pull()
    df = pd.read_csv(dir)
    df = get_final_score_quality(df, selected_columns)
    df = df.sort_values(by="Selected Score", ascending=False)
    return df

def get_all_df_i2v(selected_columns, dir=I2V_DIR):
    # submission_repo = Repository(local_dir=SUBMISSION_NAME, clone_from=SUBMISSION_URL, use_auth_token=HF_TOKEN, repo_type="dataset")
    # submission_repo.git_pull()
    df = pd.read_csv(dir)
    df = get_final_score_i2v(df, selected_columns)
    df = df.sort_values(by="Selected Score", ascending=False)
    return df

def get_all_df_long(selected_columns, dir=LONG_DIR):
    submission_repo = Repository(local_dir=SUBMISSION_NAME, clone_from=SUBMISSION_URL, use_auth_token=HF_TOKEN, repo_type="dataset")
    submission_repo.git_pull()
    df = pd.read_csv(dir)
    df = get_final_score(df, selected_columns)
    df = df.sort_values(by="Selected Score", ascending=False)
    return df


def convert_scores_to_percentage(df):
    # 对DataFrame中的每一列（除了'name'列）进行操作

    if 'Source' in df.columns:
        skip_col =2
    else:
        skip_col =1
    for column in df.columns[skip_col:]:  # 假设第一列是'name'
        df[column] = round(df[column] * 100,2)  # 将分数转换为百分数
        df[column] = df[column].apply(lambda x:  f"{x:05.2f}") + '%'
    return df

def choose_all_quailty():
    return gr.update(value=QUALITY_LIST)

def choose_all_semantic():
    return gr.update(value=SEMANTIC_LIST)

def disable_all():
    return gr.update(value=[])
    
def enable_all():
    return gr.update(value=TASK_INFO)

# select function
def on_filter_model_size_method_change(selected_columns):
    updated_data = get_all_df(selected_columns, CSV_DIR)
    #print(updated_data)
    # columns:
    selected_columns = [item for item in TASK_INFO if item in selected_columns]
    present_columns = MODEL_INFO + selected_columns
    updated_data = updated_data[present_columns]
    updated_data = updated_data.sort_values(by="Selected Score", ascending=False)
    updated_data = convert_scores_to_percentage(updated_data)
    updated_headers = present_columns
    update_datatype = [DATA_TITILE_TYPE[COLUMN_NAMES.index(x)] for x in updated_headers]
    # print(updated_data,present_columns,update_datatype)
    filter_component = gr.components.Dataframe(
        value=updated_data, 
        headers=updated_headers,
        type="pandas", 
        datatype=update_datatype,
        interactive=False,
        visible=True,
        )
    return filter_component#.value

def on_filter_model_size_method_change_quality(selected_columns):
    updated_data = get_all_df_quality(selected_columns, QUALITY_DIR)
    #print(updated_data)
    # columns:
    selected_columns = [item for item in QUALITY_TAB if item in selected_columns]
    present_columns = MODEL_INFO_TAB_QUALITY + selected_columns
    updated_data = updated_data[present_columns]
    updated_data = updated_data.sort_values(by="Selected Score", ascending=False)
    updated_data = convert_scores_to_percentage(updated_data)
    updated_headers = present_columns
    update_datatype = [DATA_TITILE_TYPE[COLUMN_NAMES.index(x)] for x in updated_headers]
    # print(updated_data,present_columns,update_datatype)
    filter_component = gr.components.Dataframe(
        value=updated_data, 
        headers=updated_headers,
        type="pandas", 
        datatype=update_datatype,
        interactive=False,
        visible=True,
        )
    return filter_component#.value

def on_filter_model_size_method_change_i2v(selected_columns):
    updated_data = get_all_df_i2v(selected_columns, I2V_DIR)
    selected_columns = [item for item in I2V_TAB if item in selected_columns]
    present_columns = MODEL_INFO_TAB_I2V + selected_columns
    updated_data = updated_data[present_columns]
    updated_data = updated_data.sort_values(by="Selected Score", ascending=False)
    updated_data = convert_scores_to_percentage(updated_data)
    updated_headers = present_columns
    update_datatype = [DATA_TITILE_TYPE[COLUMN_NAMES_I2V.index(x)] for x in updated_headers]
    # print(updated_data,present_columns,update_datatype)
    filter_component = gr.components.Dataframe(
        value=updated_data, 
        headers=updated_headers,
        type="pandas", 
        datatype=update_datatype,
        interactive=False,
        visible=True,
        )
    return filter_component#.value

def on_filter_model_size_method_change_long(selected_columns):
    updated_data = get_all_df_long(selected_columns, LONG_DIR)
    selected_columns = [item for item in TASK_INFO if item in selected_columns]
    present_columns = MODEL_INFO + selected_columns
    updated_data = updated_data[present_columns]
    updated_data = updated_data.sort_values(by="Selected Score", ascending=False)
    updated_data = convert_scores_to_percentage(updated_data)
    updated_headers = present_columns
    update_datatype = [DATA_TITILE_TYPE[COLUMN_NAMES.index(x)] for x in updated_headers]
    filter_component = gr.components.Dataframe(
        value=updated_data, 
        headers=updated_headers,
        type="pandas", 
        datatype=update_datatype,
        interactive=False,
        visible=True,
        )
    return filter_component#.value

block = gr.Blocks()


with block:
    gr.Markdown(
        LEADERBORAD_INTRODUCTION
    )
    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        # Table 0
        with gr.TabItem("📊 VBench", elem_id="vbench-tab-table", id=1):
            with gr.Row():
                with gr.Accordion("Citation", open=False):
                    citation_button = gr.Textbox(
                        value=CITATION_BUTTON_TEXT,
                        label=CITATION_BUTTON_LABEL,
                        elem_id="citation-button",
                        lines=14,
                    )
    
            gr.Markdown(
                TABLE_INTRODUCTION
            )
            with gr.Row():
                with gr.Column(scale=0.2):
                    choosen_q = gr.Button("Select Quality Dimensions")
                    choosen_s = gr.Button("Select Semantic Dimensions")
                    # enable_b = gr.Button("Select All")
                    disable_b = gr.Button("Deselect All")

                with gr.Column(scale=0.8):
                    # selection for column part:
                    checkbox_group = gr.CheckboxGroup(
                        choices=TASK_INFO,
                        value=DEFAULT_INFO,
                        label="Evaluation Dimension",
                        interactive=True,
                    )

            data_component = gr.components.Dataframe(
                value=get_baseline_df, 
                headers=COLUMN_NAMES,
                type="pandas", 
                datatype=DATA_TITILE_TYPE,
                interactive=False,
                visible=True,
                height=700,
                )
    
            choosen_q.click(choose_all_quailty, inputs=None, outputs=[checkbox_group]).then(fn=on_filter_model_size_method_change, inputs=[ checkbox_group], outputs=data_component)
            choosen_s.click(choose_all_semantic, inputs=None, outputs=[checkbox_group]).then(fn=on_filter_model_size_method_change, inputs=[ checkbox_group], outputs=data_component)
            # enable_b.click(enable_all, inputs=None, outputs=[checkbox_group]).then(fn=on_filter_model_size_method_change, inputs=[ checkbox_group], outputs=data_component)
            disable_b.click(disable_all, inputs=None, outputs=[checkbox_group]).then(fn=on_filter_model_size_method_change, inputs=[ checkbox_group], outputs=data_component)
            checkbox_group.change(fn=on_filter_model_size_method_change, inputs=[ checkbox_group], outputs=data_component)

        # Table 1
        with gr.TabItem("Video Quaity", elem_id="vbench-tab-table", id=2):
            with gr.Accordion("INSTRUCTION", open=False):
                    citation_button = gr.Textbox(
                        value=QUALITY_CLAIM_TEXT,
                        label="",
                        elem_id="quality-button",
                        lines=2,
                    )
            with gr.Row():
                with gr.Column(scale=1.0):
                    # selection for column part:
                    checkbox_group_quality = gr.CheckboxGroup(
                        choices=QUALITY_TAB,
                        value=QUALITY_TAB,
                        label="Evaluation Quality Dimension",
                        interactive=True,
                    )

            data_component_quality = gr.components.Dataframe(
                value=get_baseline_df_quality, 
                headers=COLUMN_NAMES_QUALITY,
                type="pandas", 
                datatype=DATA_TITILE_TYPE,
                interactive=False,
                visible=True,
                )
    
            checkbox_group_quality.change(fn=on_filter_model_size_method_change_quality, inputs=[checkbox_group_quality], outputs=data_component_quality)
        
        # Table i2v
        with gr.TabItem("VBench-I2V", elem_id="vbench-tab-table", id=3):
            with gr.Accordion("NOTE", open=False):
                    i2v_note_button = gr.Textbox(
                        value=I2V_CLAIM_TEXT,
                        label="",
                        elem_id="quality-button",
                        lines=3,
                    )
            with gr.Row():
                with gr.Column(scale=1.0):
                    # selection for column part:
                    checkbox_group_i2v = gr.CheckboxGroup(
                        choices=I2V_TAB,
                        value=I2V_TAB,
                        label="Evaluation Quality Dimension",
                        interactive=True,
                    )

            data_component_i2v = gr.components.Dataframe(
                value=get_baseline_df_i2v, 
                headers=COLUMN_NAMES_I2V,
                type="pandas", 
                datatype=I2V_TITILE_TYPE,
                interactive=False,
                visible=True,
                )
    
            checkbox_group_i2v.change(fn=on_filter_model_size_method_change_i2v, inputs=[checkbox_group_i2v], outputs=data_component_i2v)
        
        with gr.TabItem("📊 VBench-Long", elem_id="vbench-tab-table", id=4):
            with gr.Row():
                with gr.Accordion("INSTRUCTION", open=False):
                    citation_button = gr.Textbox(
                        value=LONG_CLAIM_TEXT,
                        label="",
                        elem_id="long-ins-button",
                        lines=2,
                    )
    
            gr.Markdown(
                TABLE_INTRODUCTION
            )
            with gr.Row():
                with gr.Column(scale=0.2):
                    choosen_q_long = gr.Button("Select Quality Dimensions")
                    choosen_s_long = gr.Button("Select Semantic Dimensions")
                    enable_b_long = gr.Button("Select All")
                    disable_b_long = gr.Button("Deselect All")

                with gr.Column(scale=0.8):
                    checkbox_group_long = gr.CheckboxGroup(
                        choices=TASK_INFO,
                        value=DEFAULT_INFO,
                        label="Evaluation Dimension",
                        interactive=True,
                    )

            data_component = gr.components.Dataframe(
                value=get_baseline_df_long, 
                headers=COLUMN_NAMES,
                type="pandas", 
                datatype=DATA_TITILE_TYPE,
                interactive=False,
                visible=True,
                height=700,
                )
    
            choosen_q_long.click(choose_all_quailty, inputs=None, outputs=[checkbox_group_long]).then(fn=on_filter_model_size_method_change_long, inputs=[ checkbox_group_long], outputs=data_component)
            choosen_s_long.click(choose_all_semantic, inputs=None, outputs=[checkbox_group_long]).then(fn=on_filter_model_size_method_change_long, inputs=[ checkbox_group_long], outputs=data_component)
            enable_b_long.click(enable_all, inputs=None, outputs=[checkbox_group_long]).then(fn=on_filter_model_size_method_change_long, inputs=[ checkbox_group_long], outputs=data_component)
            disable_b_long.click(disable_all, inputs=None, outputs=[checkbox_group_long]).then(fn=on_filter_model_size_method_change_long, inputs=[ checkbox_group_long], outputs=data_component)
            checkbox_group_long.change(fn=on_filter_model_size_method_change_long, inputs=[checkbox_group_long], outputs=data_component)
            
        # table info
        with gr.TabItem("📝 About", elem_id="mvbench-tab-table", id=5):
            gr.Markdown(LEADERBORAD_INFO, elem_classes="markdown-text")
        
        # table submission 
        with gr.TabItem("🚀 Submit here! ", elem_id="mvbench-tab-table", id=6):
            gr.Markdown(LEADERBORAD_INTRODUCTION, elem_classes="markdown-text")

            with gr.Row():
                gr.Markdown(SUBMIT_INTRODUCTION, elem_classes="markdown-text")

            with gr.Row():
                gr.Markdown("# ✉️✨ Submit your model evaluation json file here!", elem_classes="markdown-text")

            with gr.Row():
                with gr.Column():
                    model_name_textbox = gr.Textbox(
                        label="**Model name**", placeholder="Required field"
                        )
                    revision_name_textbox = gr.Textbox(
                        label="Revision Model Name(Optional)", placeholder="LaVie"
                    )

                with gr.Column():
                    model_link = gr.Textbox(
                        label="**Project Page/Paper Link**", placeholder="Required field"
                    )
                    team_name = gr.Textbox(
                        label="Your Team Name(If left blank, it will be user upload)", placeholder="User Upload"
                    )
                    contact_email = gr.Textbox(
                        label="E-Mail(**Will not be displayed**)", placeholder="Required field"
                    )


            with gr.Column():

                input_file = gr.components.File(label = "Click to Upload a ZIP File", file_count="single", type='binary')
                submit_button = gr.Button("Submit Eval")
                submit_succ_button = gr.Markdown("Submit Success! Please press refresh and return to LeaderBoard!", visible=False)
                fail_textbox = gr.Markdown('<span style="color:red;">Please ensure that the `Model Name`, `Project Page`, and `Email` are filled in correctly.</span>', elem_classes="markdown-text",visible=False)
                
    
                submission_result = gr.Markdown()
                submit_button.click(
                    add_new_eval,
                    inputs = [
                        input_file,
                        model_name_textbox,
                        revision_name_textbox,
                        model_link,
                        team_name,
                        contact_email
                    ],
                    outputs=[submit_button, submit_succ_button, fail_textbox]
                )


    def refresh_data():
        value1 = get_baseline_df()
        return value1

    with gr.Row():
        data_run = gr.Button("Refresh")
        data_run.click(on_filter_model_size_method_change, inputs=[checkbox_group], outputs=data_component)


block.launch()
