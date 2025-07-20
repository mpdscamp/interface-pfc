import { useEffect, useState } from 'react';
import {
  Table, TableBody, TableCell, TableContainer, TableHead,
  TableRow, Paper, LinearProgress, Link, Box
} from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';

interface Job {
  id: string;
  kind: string;
  status: string;
  progress: number;
  submitted_at: string;
  metrics_json?: {
    current_loss?: number;
    current_epoch?: number;
    current_batch?: number;
    elapsed?: number;
    eta?: number;
    detailed_progress?: number;
  };
}

const formatTime = (secs: number) => {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  return `${h>0?h+'h ':''}${m}m ${s}s`;
};

export default function JobStatusTable() {
  const [jobs, setJobs] = useState<Job[]>([]);

  const load = () => fetch('/api/jobs')
    .then(r => r.json()).then(setJobs);

  useEffect(() => { load(); const t = setInterval(load, 2000); return () => clearInterval(t); }, []);

  const prettyKind = (k: string) =>
    k.replace('_', ' → ').replace('tabular', 'Tabular').replace('llm', 'LLM');

  return (
    <TableContainer component={Paper} sx={{ mt: 2 }}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Kind</TableCell>
            <TableCell>Status</TableCell>
            <TableCell>Progress</TableCell>
            <TableCell>Submitted At</TableCell>
            <TableCell>Result</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {jobs.map(j => (
            <TableRow key={j.id}>
              <TableCell>{prettyKind(j.kind)}</TableCell>
              <TableCell>{j.status}</TableCell>
              <TableCell sx={{ minWidth: 220 }}>
                {j.status === 'RUNNING' ? (
                  <>
                    <LinearProgress variant="determinate" value={j.progress} />
                    <Box sx={{ mt: 1, fontSize: '0.875rem' }}>
                      {/* two‑decimals % */}
                      {(j.metrics_json?.detailed_progress ?? j.progress).toFixed(2)}%
                      {/* current loss */}
                      {j.metrics_json?.current_loss != null &&
                        ` | loss: ${j.metrics_json.current_loss.toFixed(4)}`}
                      {/* elapsed / ETA */}
                      {j.metrics_json?.elapsed != null && j.metrics_json?.eta != null && (
                        ` | elapsed: ${formatTime(j.metrics_json.elapsed)}`
                      )}
                      {j.metrics_json?.elapsed != null && j.metrics_json?.eta != null && (
                        ` | ETA: ${formatTime(j.metrics_json.eta)}`
                      )}
                    </Box>
                  </>
                ) : (
                  j.progress === 100 ? 'Done' : `${j.progress}%`
                )}
              </TableCell>
              <TableCell>{new Date(j.submitted_at).toLocaleString()}</TableCell>
              <TableCell>
                {j.kind.endsWith('_infer') && j.status === 'COMPLETED' && (
                  <Link component={RouterLink} to={`/results/${j.id}`}>View</Link>
                )}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
