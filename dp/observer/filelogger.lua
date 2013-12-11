------------------------------------------------------------------------
--[[ FileLogger ]]--
-- Interface, Observer
-- Simple logger that prints a report every epoch
------------------------------------------------------------------------

local FileLogger, parent = torch.class("dp.FileLogger", "dp.Logger")
FileLogger.isFileLogger = true

function FileLogger:__init(save_dir)
   self._save_dir = save_dir or dp.SAVE_DIR
   parent.__init(self)
end

function FileLogger:setup(config)
   parent.setup(self, config)
   --concatenate save directory with subject id
   local subject_path = self._subject:id():toPath()
   self._save_dir = paths.concat(self._save_dir, subject_path)
   self._log_dir = paths.concat(self._save_dir, 'log')
   --creates directories if required
   os.execute('mkdir -p ' .. sys.dirname(self._log_dir))
end

function FileLogger:doneEpoch(report)
   self._last_epoch = report.epoch
   local filename = paths.concat(self._log_dir, 
                                 'report_' .. report.epoch .. '.dat')
   torch.save(filename, report)
   filename = paths.concat(self._log_dir, 'metadata.dat')
   torch.save(filename, self._last_epoch)
end

function FileLogger:logHyperReport(hyper_report)
   local filename = paths.concat(self._log_dir, 'hyper_report.dat')
   torch.save(filename, hyper_report)
end
