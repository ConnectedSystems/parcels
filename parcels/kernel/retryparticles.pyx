from parcels.tools.loggers import logger
from parcels.tools.statuscodes import OperationCode, StateCode
from parcels.tools.statuscodes import recovery_map as recovery_base_map


cpdef retry_particles(cls, pset, output_file, endtime, dt):
    # Identify particles that threw errors
    cdef int n_error = pset.num_error_particles
    while n_error > 0:
        error_pset = pset.error_particles
        # Apply recovery kernel
        for p in error_pset:
            if p.state == OperationCode.StopExecution:
                return
            if p.state == OperationCode.Repeat:
                p.reset_state()
            elif p.state == OperationCode.Delete:
                pass
            elif p.state in recovery_base_map:
                recovery_kernel = recovery_base_map[p.state]
                p.set_state(StateCode.Success)
                recovery_kernel(p, cls.fieldset, p.time)
                if p.isComputed():
                    p.reset_state()
            else:
                logger.warning_once(
                    f"Deleting particle {p.id} because of non-recoverable error"
                )
                p.delete()

        # Remove all particles that signalled deletion
        cls.remove_deleted(
            pset, output_file=output_file, endtime=endtime
        )  # Generalizable version!

        # Execute core loop again to continue interrupted particles
        if cls.ptype.uses_jit:
            cls.execute_jit(pset, endtime, dt)
        else:
            cls.execute_python(pset, endtime, dt)

        n_error = pset.num_error_particles
