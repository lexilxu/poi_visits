"""Compile with an existing MiKTeX installation and project-local caches.

No package installation, registry update, shell escape, or publication is
performed. The runtime and build directories must be writable by the caller.
"""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json
import os
import shutil
import subprocess


def compile_document(args):
    main=args.main.resolve();runtime=args.runtime.resolve();out=args.out.resolve()
    if not main.is_file():raise FileNotFoundError(main)
    binary=args.miktex_bin
    if binary is None:
        found=shutil.which('pdflatex')
        if not found:raise FileNotFoundError('An existing MiKTeX installation is required')
        binary=Path(found).parent
    binary=binary.resolve();install=binary.parents[2]
    required=['pdflatex.exe','miktex-pdftex.exe','miktex-bibtex.exe']
    if any(not (binary/name).is_file() for name in required):
        raise FileNotFoundError('The selected directory is not a complete MiKTeX binary directory')
    for path in (runtime,runtime/'config',runtime/'data',out):path.mkdir(parents=True,exist_ok=True)
    startup=runtime/'miktexstartup.ini'
    lines=['[Auto]','Config=Portable','[Setup]','Version=2.9','[Paths]']
    for scope in ('User','Common'):
        for name,path in [('Config',runtime/'config'),('Data',runtime/'data'),('Install',install)]:
            lines.append(f'{scope}{name}={path}')
    startup.write_text('\n'.join(lines)+'\n',encoding='utf8')
    env={**os.environ,'MIKTEX_COMMONSTARTUPFILE':str(startup),'MIKTEX_USERSTARTUPFILE':str(startup),
         'MIKTEX_USERCONFIG':str(runtime/'config'),'MIKTEX_USERDATA':str(runtime/'data'),
         'MIKTEX_USERINSTALL':str(install),'BIBINPUTS':str(main.parent)+os.pathsep}
    commands=[]
    def run(name,command,cwd):
        commands.append(command)
        with (out/(name+'.console.log')).open('w',encoding='utf8') as log:
            result=subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        if result.returncode:
            log=(out/(name+'.console.log')).read_text(encoding='utf8',errors='replace')
            raise RuntimeError(f'{name} failed:\n'+log[-3500:])
    common=['--disable-installer','--miktex-disable-maintenance','--miktex-disable-diagnose',
            '-interaction=nonstopmode','-halt-on-error','-no-shell-escape']
    fmt=runtime/'pdflatex.fmt'
    if not fmt.exists():
        run('format',[str(binary/'miktex-pdftex.exe')]+common+[
            '--initialize','--alias=pdflatex','--job-name=pdflatex','--enable-etex',
            '-output-directory='+str(runtime),'pdflatex.ini'],runtime)
    command=[str(binary/'pdflatex.exe')]+common+['-fmt='+str(fmt),'-output-directory='+str(out),main.name]
    run('latex_1',command,main.parent)
    aux=out/(main.stem+'.aux')
    if '\\bibdata' in aux.read_text(encoding='utf8'):
        run('bibtex',[str(binary/'miktex-bibtex.exe'),'--disable-installer',main.stem],out)
    run('latex_2',command,main.parent)
    run('latex_3',command,main.parent)
    for pass_number in range(4,7):
        current_log=(out/(main.stem+'.log')).read_text(encoding='utf8',errors='replace')
        if 'Label(s) may have changed' not in current_log and 'Rerun to get cross-references right' not in current_log:
            break
        run(f'latex_{pass_number}',command,main.parent)
    pdf=out/(main.stem+'.pdf');log=(out/(main.stem+'.log')).read_text(encoding='utf8',errors='replace')
    if not pdf.is_file():raise RuntimeError('Compiler did not create a PDF')
    issues=[line for line in log.splitlines() if any(token in line for token in
            ('Warning:','Overfull','Underfull','undefined references','undefined citations'))]
    report=dict(status='compiled_with_messages' if issues else 'compiled',main=str(main),pdf=str(pdf),
        pdf_sha256=sha256(pdf.read_bytes()).hexdigest(),main_sha256=sha256(main.read_bytes()).hexdigest(),
        messages=issues,commands=commands,completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Compilation only; result completeness and visual inspection require separate verification.')
    (out/'build_manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:report[k] for k in ('status','pdf','pdf_sha256','messages')}))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--main',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--miktex-bin',type=Path)
    compile_document(p.parse_args())
