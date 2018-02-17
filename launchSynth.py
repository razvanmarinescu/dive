from launchCommon import *


parser = argparse.ArgumentParser(description='Launches freesurfer processes for ADNI/DRC thickness data on cluster or local machine')
addParserArgs(parser)

parser.add_argument('--expToRun', dest='expToRun', type=int,
                   help='index of experiment to run: 0. all 1. vary clusters 2. vary subjects')

parser.add_argument('--steps', dest='steps',
                   help='list of simulation steps to run, comma-separated e.g: 2,4,8. '
                        'For example, in the experiments with varying numbers of clusters, '
                        'each step describes one such experiment.')


# parser.add_argument('--trajFunc', dest="trajFunc", help='lin or sig')


args = parser.parse_args()

launchParams = initCommonLaunchParams(args)

def getRunCmd(model, runIndex, stepToRun, nrClust, agg):
  """

  :param model:
  :param runIndex:
  :param stepToRun:
  :param nrClust:
  :param agg: if 1 then plot without Xwindows (for cluster)
              if 0 then plot with Xwindows (for personal machine)
  :return:
  runCmdList command representation as a list of pairs
  runCmdStr command representatition as a string
  """
  runCmdList = [('python3', args.launcherScript), ('--runIndex', runIndex),
    ('--nrProc',  args.nrProc),
    ('--modelToRun', model), ('--expToRun', args.expToRun), ('--stepToRun', stepToRun),
    ('--nrOuterIt', args.nrOuterIt),
    ('--nrInnerIt', args.nrInnerIt), ('--nrClust', nrClust),
    ('--initClustering',  args.initClustering), ('--agg', agg),
    ('--rangeFactor', '%.2f' % args.rangeFactor), ('--informPrior', args.informPrior)]

  runCmdStr = ' '.join(['%s %s' % (s,str(a)) for (s,a) in runCmdList])

  return runCmdList, runCmdStr

def getQsubCmd(model, runIndex, stepToRun, nrClust):
  ''' Creates full command for cluster'''
  # if there's an error about tty, add & in the last parameter
  runCmd = ('cd %s; /share/apps/python-3.5.1/bin/' % launchParams['REPO_DIR']) + \
           getRunCmd(model,runIndex, stepToRun, nrClust, agg=1)[1]
  runCmd += ' --cluster '
  jobName = "i%d_m%d_c%d_%s_%s_r%.1f_p%d" % (runIndex, model,nrClust,
               args.launcherScript.split('.')[0], args.initClustering,args.rangeFactor,
               args.informPrior)
  qsubCmd = getQsubCmdPart2(launchParams, jobName)
  cmdClust = 'echo "%s" | %s' % (runCmd, qsubCmd) # echo the matlab cmd then pipe it to qsub
  return cmdClust, runCmd

pList = []
#instanceIndices = [2,3,4,5,6,7,8];
instanceIndices = range(args.firstInstance, args.lastInstance+1)
quitFlag = 1

if args.firstModel and args.lastModel:
  modelIndices = range(args.firstModel, args.lastModel + 1)
elif args.models:
  modelIndices = [int(i) for i in args.models.split(',')]
else:
  raise ValueError('need to set either --models or --firstModel & --lastModel')

if args.nrClust:
  nrClustList = [args.nrClust]
elif args.nrClustList:
  nrClustList = [int(i) for i in args.nrClustList.split(',')]
else:
  raise ValueError('set either --nrClust or --nrClustList')

stepIndices = [int(i) for i in args.steps.split(',')]

# cmdArgsList = [('python3 ', args.launcherScript), ('--nrProc')]

for m in modelIndices:
  for i in instanceIndices:
    for s in stepIndices:
      for c in nrClustList:
        if not args.cluster:
          if args.serial:
            _, cmd = getRunCmd(args.models, i, s, c, agg=0)
            os.system(cmd)
          else:
            cmdArgs,cmd = getRunCmd(m, i, s, c, agg=0)
            print(cmdArgs)
            p = subprocess.Popen(cmd.split(' '))
            pList.append(p)
        else:
          # run on cluster
          cmdClust, runCmd = getQsubCmd(m, i, s, c)
          print(cmdClust)
          if not args.printOnly:
            os.system(cmdClust)

# if I launch processes on local machine, wait for them to finish.
if not args.cluster and not args.serial:
  nrProcs = len(pList)
  for i in range(nrProcs):
    p = pList.pop()
    print(p)
    p.wait()

    print("---------------------->finished")