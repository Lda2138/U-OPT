"""Validate all seven final reports and recompute per-object arithmetic means."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

GROUPS = ('U','G','T','GT','GS','TS','GTS')


def summarize(stage, output):
    prep_path = stage/'preparation.json'
    prep = json.loads(prep_path.read_text(encoding='utf-8'))
    fingerprint = hashlib.sha256(prep_path.read_bytes()).hexdigest()
    objects = prep['objects']
    if len(objects)!=1356 or len(set(objects))!=1356:
        raise ValueError('Expected1356 unique test objects')
    all_rows, summaries, paired = [], [], {}
    for group in GROUPS:
        report = json.loads((stage/('final_'+group+'.json')).read_text(encoding='utf-8'))
        if (report.get('all_passed') is not True or report.get('mode')!='full' or report.get('groups')!=[group] or
            report.get('evaluation_fingerprint')!=fingerprint or report.get('protocol')!=prep['protocol']):
            raise ValueError('Failed, incomplete or mismatched report: '+group)
        if report.get('checkpoint_sources',{}).get(group)!=prep['checkpoints'][group]:
            raise ValueError('Checkpoint metadata mismatch: '+group)
        rows = report['rows']
        if len(rows)!=1356 or [r['object_id'] for r in rows]!=objects or any(r['group']!=group for r in rows):
            raise ValueError('Object count/order mismatch: '+group)
        if report['new_objects_evaluated']+report['objects_reused']!=len(rows):
            raise ValueError('New/reused counts mismatch')
        for row in rows:
            if any(not math.isfinite(v) for v in row.values() if isinstance(v,(int,float))):
                raise ValueError('Nonfinite metric')
            identity = tuple(row[k] for k in ('view','seed','gt_available','gt_used','gt_subset_sha256','pred_points','gt_diagonal'))
            if row['object_id'] in paired and identity!=paired[row['object_id']]:
                raise ValueError('Unpaired test data/seed')
            paired[row['object_id']] = identity
        keys = [key for key,value in rows[0].items() if key.startswith(('raw_','icp_','mesh_','image_')) and isinstance(value,(int,float))]
        means = {key:math.fsum(row[key] for row in rows)/len(rows) for key in keys}
        for key,value in report['metric_means'][group].items():
            if not math.isclose(means[key],value,rel_tol=2e-12,abs_tol=2e-12):
                raise ValueError('Reported mean differs from object results: '+key)
        summaries.append({'group':group,**means})
        all_rows.extend(rows)
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in (('seven_group_metrics.csv',summaries),('all_objects.csv',all_rows)):
        with (output/name).open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    summary={'evaluation_fingerprint':fingerprint,'groups':len(summaries),'rows':len(all_rows),'means':summaries,
             'scope':'Single-seed arithmetic means; no inference rerun and no statistical significance claim.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('ACCEPTED: 7 groups,9492 paired records; summaries written to '+str(output))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summarize(args.stage,args.output)


if __name__=='__main__':main()
