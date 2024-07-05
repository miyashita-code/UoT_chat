def get_task(args):
    if args.task == "dementia-support":
        from uot.tasks.dementia_support import DementiaSupportTask
        return DementiaSupportTask(args)
    else:
        raise NotImplementedError

