from threading import Lock

import py_vncorenlp


class VnCoreNLP_Singleton:
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls, save_dir):
        with cls._lock:
            if cls._instance is None:
                cls._instance = py_vncorenlp.VnCoreNLP(
                    # annotators=["wseg"],
                    save_dir=save_dir
                )
            return cls._instance
