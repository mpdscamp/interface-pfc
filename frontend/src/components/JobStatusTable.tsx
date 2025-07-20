import { useEffect, useState } from 'react';
import {
  Table, TableBody, TableCell, TableContainer, TableHead,
  TableRow, Paper, LinearProgress, Link
} from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';

interface Job {
  id: string;
  kind: string;               // tabular_train, llm_infer, …
  status: string;
  progress: number;
  submitted_at: string;
}

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
              <TableCell sx={{ minWidth: 150 }}>
                {j.status === 'RUNNING'
                  ? <LinearProgress variant="determinate" value={j.progress} />
                  : (j.progress === 100 ? 'Done' : `${j.progress}%`)}
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
