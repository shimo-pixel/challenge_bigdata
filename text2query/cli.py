from __future__ import annotations

import argparse
import json
from pathlib import Path
from .io import dump_json
from .pipeline import Pipeline, validate_submission


def parser():
    root=argparse.ArgumentParser(prog='text2query')
    commands=root.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare')
    prepare.add_argument('--archive',type=Path,required=True)
    prepare.add_argument('--administrative',type=Path)
    prepare.add_argument('--workdir',type=Path,default=Path('.t2q'))
    for name in ('predict','evaluate','validate'):
        p=commands.add_parser(name)
        p.add_argument('--workdir',type=Path,default=Path('.t2q'))
    predict=commands.choices['predict']
    predict.add_argument('--output',type=Path,required=True)
    predict.add_argument('--offline',action='store_true',help='Copy baseline only, not recommended for submissions')
    predict.add_argument('--no-review',action='store_true')
    predict.add_argument('--no-resume',action='store_true')
    predict.add_argument('--workers',type=int,default=4)
    predict.add_argument('--limit',type=int,help='Debug subset; cannot be submitted')
    evaluate=commands.choices['evaluate']
    evaluate.add_argument('--limit',type=int)
    evaluate.add_argument('--model',action='store_true')
    evaluate.add_argument('--workers',type=int,default=4)
    evaluate.add_argument('--report',type=Path,default=Path('reports/evaluation.json'))
    commands.choices['validate'].add_argument('--input',type=Path,required=True)
    return root


def main():
    args=parser().parse_args()
    if args.command=='prepare':
        Pipeline.prepare(args.archive,args.workdir,args.administrative)
        print(f'prepared {args.workdir}')
    elif args.command=='validate':
        result=validate_submission(args.workdir,args.input)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if not result['ok']: raise SystemExit(1)
    else:
        pipeline=Pipeline(args.workdir)
        if args.command=='evaluate':
            result=pipeline.evaluate(args.limit,args.model,args.workers,args.report)
            compact={k:v for k,v in result.items() if k not in {'baseline','generated','reviewed'}}
            for key in ('baseline','generated','reviewed'):
                if key in result: compact[key]=result[key]['metrics']
            print(json.dumps(compact,ensure_ascii=False,indent=2))
        else:
            rows=pipeline.test[:args.limit] if args.limit else None
            result=pipeline.predict(args.offline,not args.no_review,args.workers,not args.no_resume,rows)
            dump_json(args.output,result)
            print(f'wrote {args.output}; usage={json.dumps(pipeline.generator.stats)}')

if __name__=='__main__': main()
