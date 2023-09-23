# project.py


import pandas as pd
import numpy as np
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    sylla_dict = {
        'lab': [],
        'project': [],
        'midterm': [],
        'final': [],
        'disc': [],
        'checkpoint': []
    }
    
    
    for col in grades.columns:
        col = col.split()[0]
        if 'lab' in col:
            if col not in sylla_dict['lab']:
                sylla_dict['lab'].append(col)
        elif 'Midterm' in col:
            if col not in sylla_dict['midterm']:
                sylla_dict['midterm'].append(col)
        elif 'Final' in col:
            if col not in sylla_dict['final']:
                sylla_dict['final'].append(col)
        elif 'disc' in col:
            if col not in sylla_dict['disc']:
                sylla_dict['disc'].append(col)
        elif 'checkpoint' in col:
            clean = col.split('_')[1]
            if clean not in sylla_dict['checkpoint']:
                sylla_dict['checkpoint'].append(clean)                
        elif 'project' in col:
            clean = col.split('_')[0]
            if clean not in sylla_dict['project']:
                sylla_dict['project'].append(clean)
        
    for assn in sylla_dict:
        sylla_dict[assn].sort()
    
    return sylla_dict
    

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades):
    
    proj_dict = {}
    for col in grades.columns:
        if 'project' in col:
            if '_' in col:
                clean = col.split('_')[0].strip()
                proj_dict[clean] = 100
             
    dict_keys = list(proj_dict.keys())
    dict_keys.sort()
    sorted_proj_dict = {i: proj_dict [i] for i in dict_keys}

    proj_grades = grades.filter(like='project')

    zero_fill = len(str(len(sorted_proj_dict.keys())))+1

    def sussy (dic):
        result = np.zeros(grades.shape[0])
        for i in range(1,len(proj_dict.keys())+1):
            grade = f'project{str(i).zfill(zero_fill)}'
            frq = f'project{str(i).zfill(zero_fill)}_free_response'
            grade_max = f'project{str(i).zfill(zero_fill)} - Max Points'
            frq_max = f'project{str(i).zfill(zero_fill)}_free_response - Max Points'
            if str(frq) in grades.columns:
                grade_calc = ((proj_grades[str(grade)] + proj_grades[str(frq)]) / (proj_grades[str(grade_max)] + proj_grades[str(frq_max)]))
            else:
                grade_calc = (proj_grades[str(grade)] / proj_grades[str(grade_max)])
            result += np.array(grade_calc)
            
        return result / (len(proj_dict.keys()))
    series = pd.Series(sussy(proj_grades))
    return series.replace(np.nan, 0)
    

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def last_minute_submissions(grades):    
    
    lab_late = grades.copy().filter(like='lab')
    lab_late = lab_late.filter(like='Lateness').fillna(0)
    
    count_dict = {}
    for lab in lab_late.columns:
        lab_split = lab.split()
        lab_late[f'thresh_{lab_split[0]}'] = lab_late[f'{lab_split[0]} - Lateness (H:M:S)'] / pd.Timedelta(seconds=1)
        if 'Lateness' in lab:
            late_count = lab_late[(lab_late[f'thresh_{lab_split[0]}'] > 0) & (lab_late[f'thresh_{lab_split[0]}'] <= 28800)][f"{lab_split[0]} - Lateness (H:M:S)"].count()
            count_dict[lab_split[0]] = late_count
    return pd.Series(count_dict)

# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------

def lateness_penalty(col):
    
    hrs = col.apply(lambda x: int(x.split(':')[0]))
    mins = col.apply(lambda x: int(x.split(':')[1]))
    secs = col.apply(lambda x: int(x.split(':')[2]))

    thresh = 604800
    #lateness_sec = hrs*60*60 + mins*60  + secs
    lateness_hrs = hrs + mins/60 + secs/(60*60)
    
    
    lateness_multipliers =  lateness_hrs.apply(lambda x: 1.0 if x == 0 else \
                                    (0.9 if  x <= 168 else \
                                    (0.7 if  x <= 168*2 else \
                                    (0.4))))

    return lateness_multipliers
    

# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def process_labs(grades):
    lab_grades = grades.copy().filter(like='lab').fillna(0)

    lab_lst = []
    ser_lst = []
    fin_dict = {}
    for col in lab_grades.columns:
        if 'lab' in col:
            if '-'not in col:
                clean = col.split('_')[0].strip()
                lab_lst.append(clean)

    for lab in lab_lst:
        proc = ((lab_grades[lab] * lateness_penalty(lab_grades[f'{lab} - Lateness (H:M:S)'])) / lab_grades[f'{lab} - Max Points']) 
        #normalized = (proc - proc.min()) / (proc.max() - proc.min())
        ser_lst.append(proc)
        
    for i in range(len(lab_lst)):
        fin_dict[lab_lst[i]] = ser_lst[i]
    return pd.DataFrame(fin_dict)


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def lab_total(processed):
    drop_lst = []
    df = processed.T

    for col in df.columns: 
        ser = df[col].drop(df[col].idxmin())
        drop_lst.append(ser.mean())

    return pd.Series(drop_lst)


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def total_points(grades):
    #grade series
    lab_grades = lab_total(process_labs(grades.fillna(0)))
    proj_grades = projects_total(grades.fillna(0))
    chec_grades = 0 #done
    disc_grades = 0 #done
    mid_grades = 0 #done
    fin_grades = 0 #done
    total_grade = 0 

    #dfs
    df = grades.copy().fillna(0)
    disc_df = df.copy().filter(like='disc')
    mid_df = df.copy().filter(like='Mid')
    fin_df = df.copy().filter(like='Fin')
    chec_df = df.copy().filter(like='check')

    #disc grades
    disc_lst = get_assignment_names(grades)['disc']
    disc_grade_lst = []
    for disc in disc_lst:
        ser_dsc = (disc_df[disc] / disc_df[f'{disc} - Max Points'])
        disc_grade_lst.append(ser_dsc)
    disc_grades = sum(disc_grade_lst)/len(disc_lst)

    #mid grades
    mid_lst = get_assignment_names(grades)['midterm']
    mid_grade_lst = []
    for mid in mid_lst:
        ser_mid = (mid_df[mid] / mid_df[f'{mid} - Max Points'])
        mid_grade_lst.append(ser_mid)
    mid_grades = sum(mid_grade_lst)/len(mid_lst)

    #fin grades
    fin_lst = get_assignment_names(grades)['final']
    fin_grade_lst = []
    for fin in fin_lst:
        ser_fin = (fin_df[fin] / fin_df[f'{fin} - Max Points'])
        fin_grade_lst.append(ser_fin)
    fin_grades = sum(fin_grade_lst)/len(fin_lst)

    #check grades
    counter = 0
    proj_lst = get_assignment_names(grades)['project']
    chec_lst = get_assignment_names(grades)['checkpoint']
    chec_grade_lst = []
    for proj in proj_lst:
        for chec in chec_lst:
            if f'{proj}_{chec}' in chec_df.columns:
                counter += 1
                ser_chec = (chec_df[f'{proj}_{chec}'] / chec_df[f'{proj}_{chec} - Max Points'])
                chec_grade_lst.append(ser_chec)
    chec_grades = sum(chec_grade_lst)/counter

    total_grade = (lab_grades * 0.2) + (proj_grades * 0.3) + (chec_grades * 0.025) + (disc_grades * 0.025) + (mid_grades * 0.15) + (fin_grades * 0.3)
    return total_grade



# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------

def grade_check(grade):
    if grade >= 0.9:
        return "A"
    elif 0.8 <= grade < 0.9:
        return "B"
    elif 0.7 <= grade < 0.8:
        return "C"
    elif 0.6 <= grade < 0.7:
        return "D"
    else:
        return "F"
        
def final_grades(total):
    sus = total.copy()
    return sus.apply(grade_check)

def letter_proportions(total):
    
    a_prop = sum(final_grades(total) == 'A') / total.size
    b_prop = sum(final_grades(total) == 'B') / total.size
    c_prop = sum(final_grades(total)== 'C') / total.size
    d_prop = sum(final_grades(total) == 'D') / total.size
    f_prop = sum(final_grades(total) == 'F') / total.size
    letters = {'A': a_prop, 'B': b_prop, 'C': c_prop, 'D': d_prop, 'F': f_prop}
    return pd.Series(letters).sort_values(ascending=False)


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def raw_redemption(final_breakdown, question_numbers):
    df = final_breakdown.copy()
    total_pts = 0
    redem_clean = [x - 1 for x in question_numbers]
    for i, quest in enumerate(df.columns[1:]):
        quest_proc = float(quest.split()[2].replace('(',''))
        if i in redem_clean:  
            total_pts += quest_proc
    redem_scores = df.iloc[:, question_numbers].sum(axis=1)
    raw_redem = redem_scores / total_pts
    return pd.DataFrame({'PID': df['PID'], 'Raw Redemption Score': raw_redem})

def combine_grades(grades, raw_redemption_scores):
    combined_df = pd.merge(grades, raw_redemption_scores, on='PID', how='left')
    combined_df['Raw Redemption Score'] = combined_df['Raw Redemption Score'].fillna(0)
    return combined_df


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def z_score(ser):
    mean = ser.mean()
    std = ser.std(ddof=0)
    return (ser - mean) / std
    
def add_post_redemption(grades_combined):
    mid_df = grades_combined.copy().filter(like='Mid')
    mid_lst = get_assignment_names(grades_combined)['midterm']
    mid_grade_lst = []
    for mid in mid_lst:
        ser_mid = (mid_df[mid] / mid_df[f'{mid} - Max Points'])
        mid_grade_lst.append(ser_mid)
    mid_grades = sum(mid_grade_lst)/len(mid_lst)

    fin_df = grades_combined.copy().filter(like='Fin')
    fin_lst = get_assignment_names(grades_combined)['final']
    fin_grade_lst = []
    for fin in fin_lst:
        ser_fin = (fin_df[fin] / fin_df[f'{fin} - Max Points'])
        fin_grade_lst.append(ser_fin)
    fin_grades = sum(fin_grade_lst)/len(fin_lst)

    grades_combined = grades_combined.copy()
    grades_combined['Midterm Score Pre-Redemption'] = mid_grades
    grades_combined['Midterm Score Z-Score'] = z_score(mid_grades)
    grades_combined['Redemption Raw Score'] = fin_grades - mid_grades.fillna(0)
    grades_combined['Redemption Raw Score Z-Score'] = z_score(grades_combined['Redemption Raw Score'])

    post_redemption_min = mid_grades.mean() + 0.5 * mid_grades.std(ddof=0)
    post_redemption_max = mid_grades.mean() + mid_grades.std(ddof=0)

    grades_combined['Midterm Score Post-Redemption'] = grades_combined['Midterm Score Pre-Redemption']
    grades_combined.loc[mid_grades < post_redemption_min, 'Midterm Score Post-Redemption'] = post_redemption_min / 100
    grades_combined.loc[mid_grades > post_redemption_max, 'Midterm Score Post-Redemption'] = post_redemption_max / 100
    grades_combined.loc[mid_grades.isna(), 'Midterm Score Post-Redemption'] = grades_combined['Midterm Score Z-Score'] * grades_combined['Redemption Raw Score Z-Score'] * 0.5 + 0.5

    #updated_df = grades_combined.drop(['Midterm Score Z-Score', 'Redemption Raw Score Z-Score'], axis=1)
    return grades_combined


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def total_points_post_redemption(grades_combined):
    grades_combined=grades_combined.copy()
    grades_post_redemption = add_post_redemption(grades_combined)
    final_grades = total_points(grades_post_redemption)
    return final_grades
    
def proportion_improved(grades_combined):
    grades_pre  = grades_combined['Midterm Score Pre-Redemption'].apply(grade_check)
    grades_post = grades_combined['Midterm Score Post-Redemption'].apply(grade_check)
    num_improved = sum((grades_pre > grades_post))
    prop = num_improved / grades_combined.shape[0]
    return prop

