from threading import Lock


class VnCoreNLP_Singleton:
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls, save_dir=None, annotators=None):
        with cls._lock:
            if cls._instance is None:
                # lazy import -> safe during DAG import
                from py_vncorenlp import VnCoreNLP

                kwargs = {}
                if save_dir:
                    kwargs["save_dir"] = save_dir
                if annotators:
                    kwargs["annotators"] = annotators
                # optional: small log to detect when JVM starts
                print("Initializing VnCoreNLP JVM now (will happen in worker).")
                cls._instance = VnCoreNLP(**kwargs)
            return cls._instance
