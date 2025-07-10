import React, { useEffect, useState, useCallback } from 'react';
import {
  Container, Typography, Button, FormControl, InputLabel,
  Select, MenuItem, Box, Grid
} from '@mui/material';
import JobStatusTable from '../components/JobStatusTable';
import DatasetUploader from '../components/DatasetUploader';

interface Model { filename:string; display_name:string; kind:'tabular'|'llm' }
interface Dataset { filename:string }

export default function InferencePage() {
  const [kind, setKind] = useState<'tabular'|'llm'>('tabular');
  const [models, setModels] = useState<Model[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [checkpoint, setCheckpoint] = useState('');
  const [dataset, setDataset]       = useState('');

  const loadDatasets = useCallback(() =>
    fetch(`${import.meta.env.VITE_API_URL}/api/datasets`).then(r=>r.json()).then(setDatasets), []);

  useEffect(() => {
    fetch(`${import.meta.env.VITE_API_URL}/api/models`).then(r=>r.json()).then(setModels);
    loadDatasets();
  }, [loadDatasets]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    await fetch(`${import.meta.env.VITE_API_URL}/api/jobs/infer`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ kind, checkpoint, dataset_filename:dataset })
    });
  };

  const modelsOfKind = models.filter(m => m.kind === kind);

  return (
    <Container maxWidth="md">
      <Typography variant="h4" gutterBottom>Run Inference</Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <DatasetUploader onUploaded={loadDatasets}/>
        </Grid>

        <Grid item xs={12} md={8}>
          <Box component="form" onSubmit={submit}
               sx={{ display:'flex', flexDirection:'column', gap:2 }}>

            <FormControl fullWidth>
              <InputLabel>Model Family</InputLabel>
              <Select value={kind} label="Model Family"
                      onChange={e => { setKind(e.target.value as any); setCheckpoint(''); }}>
                <MenuItem value="tabular">Tabular</MenuItem>
                <MenuItem value="llm">LLM</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth required>
              <InputLabel>Checkpoint</InputLabel>
              <Select value={checkpoint} label="Checkpoint"
                      onChange={e => setCheckpoint(e.target.value)}>
                {modelsOfKind.map(m => (
                  <MenuItem key={m.filename} value={m.filename}>{m.display_name}</MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth required>
              <InputLabel>Dataset</InputLabel>
              <Select value={dataset} label="Dataset"
                      onChange={e => setDataset(e.target.value)}>
                {datasets.map(d => (
                  <MenuItem key={d.filename} value={d.filename}>{d.filename}</MenuItem>
                ))}
              </Select>
            </FormControl>

            <Button variant="contained" type="submit" disabled={!checkpoint||!dataset}>
              Start Inference
            </Button>
          </Box>
        </Grid>
      </Grid>

      <JobStatusTable />
    </Container>
  );
}
