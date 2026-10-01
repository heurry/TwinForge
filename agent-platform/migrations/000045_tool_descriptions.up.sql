-- Publish richer model-facing descriptions for the built-in Workspace ToolSet.
-- Existing versions remain immutable so historical Run binding snapshots stay
-- replayable.  The new Agent version points at the new ToolSet revision.
BEGIN;

DO $$
DECLARE
    item RECORD;
    enriched JSONB;
    refs JSONB := '[]'::jsonb;
    toolset_definition UUID;
    toolset_spec JSONB;
    toolset_id UUID;
    target_agent_id UUID;
    agent_spec JSONB;
BEGIN
    FOR item IN
        SELECT v.id, v.definition_id, v.version, d.tool_key, v.spec
        FROM agent_platform.tool_versions v
        JOIN agent_platform.tool_definitions d ON d.id = v.definition_id
        WHERE d.tenant_id = 'demo'
          AND d.tool_key IN (
              'workspace-list-files', 'workspace-read-file',
              'workspace-search-files', 'workspace-write-file',
              'workspace-append-file', 'workspace-edit-file',
              'workspace-run-command'
          )
          AND v.status = 'published'
          AND v.version = (
              SELECT max(v2.version)
              FROM agent_platform.tool_versions v2
              WHERE v2.definition_id = v.definition_id
                AND v2.status = 'published'
          )
    LOOP
        enriched := item.spec;
        CASE item.tool_key
        WHEN 'workspace-list-files' THEN
            enriched := jsonb_set(enriched, '{definition,description}', to_jsonb('Use to discover the current Run workspace directory structure or locate a path before read_file/write_file. Do not use it to read file contents. Start with a narrow path; set recursive=true only when necessary. If truncated=true, increase max_entries or narrow path instead of repeating the same query. path is optional and workspace-relative; include_hidden is for dotfiles only.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,path,description}', to_jsonb('Optional workspace-relative directory or file path; omit to inspect the workspace root.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,recursive,description}', to_jsonb('Whether to walk descendants; keep false for an initial shallow listing.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,max_entries,description}', to_jsonb('Maximum entries to return, from 1 to 500; increase only after a truncated result.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,include_hidden,description}', to_jsonb('Include dotfiles and internal entries only when they are relevant.'::text), true);
        WHEN 'workspace-read-file' THEN
            enriched := jsonb_set(enriched, '{definition,description}', to_jsonb('Use to inspect a known UTF-8 text file in the current Run workspace before editing, verifying, or explaining it. path is required and must be the exact workspace-relative path. Read only the needed line range; do not use this to discover unknown paths (use list_files/search_files), do not create a file to satisfy a read, and do not repeat an unchanged path/start_line/line_count range because the Runtime caches it.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,path,description}', to_jsonb('Required canonical Run-workspace-relative file path; never omit this field.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,start_line,description}', to_jsonb('Optional 1-based first line; provide it when a narrow range is sufficient.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,line_count,description}', to_jsonb('Optional positive number of lines; keep the range small and change it when more context is needed.'::text), true);
        WHEN 'workspace-search-files' THEN
            enriched := jsonb_set(enriched, '{definition,description}', to_jsonb('Use when the exact file path, symbol, or literal text location is unknown. Search literal UTF-8 text within an optional narrow workspace-relative path, then use read_file on the returned hit. query is required; do not use this as a substitute for reading a known file or repeat an identical search.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,query,description}', to_jsonb('Required non-empty literal text or symbol to find; this is not a shell command.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,path,description}', to_jsonb('Optional workspace-relative directory/file scope; narrow it when known.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,max_results,description}', to_jsonb('Optional result cap from 1 to 200; use a narrow path rather than an excessive cap.'::text), true);
        WHEN 'workspace-write-file' THEN
            enriched := jsonb_set(enriched, '{definition,description}', to_jsonb('Use to create a new UTF-8 text file or intentionally replace an existing file in the current Run workspace. This is the first write for a path: emit the canonical workspace-relative path and one coherent content chunk of at most 8192 characters. For a larger source file, keep the requirement intact and continue with append_file; do not put the whole module into one call. For a small surgical change to an existing file, use edit_file instead. If the path already exists, read it first or set overwrite=true only when replacement is intentional. expected_sha256 is a file hash for optimistic concurrency, not an Artifact hash.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,path,description}', to_jsonb('Required canonical Run-workspace-relative target path; emit this before content.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,content,description}', to_jsonb('Required first coherent UTF-8 chunk; maximum 8192 characters. Continue with append_file for the remainder.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,overwrite,description}', to_jsonb('Set true only when intentionally replacing an existing file; otherwise omit or use false.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,expected_sha256,description}', to_jsonb('Optional current file hash used for optimistic concurrency; do not confuse it with an Artifact hash.'::text), true);
        WHEN 'workspace-append-file' THEN
            enriched := jsonb_set(enriched, '{definition,description}', to_jsonb('Use only to continue a file already created by write_file or an earlier append_file. The target must already exist. Append one coherent UTF-8 chunk of at most 8192 characters, using the returned tail and file hash to continue exactly. Do not use this to create a missing file, overwrite existing content, or repeat a chunk. For a precise existing-text replacement use edit_file. expected_sha256 is the current file hash, not an Artifact hash.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,path,description}', to_jsonb('Required existing canonical Run-workspace-relative file path; emit this before content.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,content,description}', to_jsonb('Required next coherent UTF-8 chunk; maximum 8192 characters. Preserve source order and do not duplicate the previous tail.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,expected_sha256,description}', to_jsonb('Optional current file hash returned by the previous write/append; detects stale concurrent state.'::text), true);
        WHEN 'workspace-edit-file' THEN
            enriched := jsonb_set(enriched, '{definition,description}', to_jsonb('Use for a small, precise replacement inside an existing UTF-8 text file when the current unique anchor is known. Read the current file/range first. old_text must match exactly once; new_text is the replacement. Do not use this to create a file, append a large new module, or guess stale content. If the anchor is missing or ambiguous, reread and choose a narrower unique anchor. expected_sha256 protects against concurrent changes.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,path,description}', to_jsonb('Required canonical existing Run-workspace-relative file path.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,old_text,description}', to_jsonb('Required non-empty exact current text that occurs uniquely in the file; do not paraphrase or use stale content.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,new_text,description}', to_jsonb('Replacement text; keep it limited to the intended surgical change.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,expected_sha256,description}', to_jsonb('Optional file hash observed during the preceding read; rejects stale edits.'::text), true);
        WHEN 'workspace-run-command' THEN
            enriched := jsonb_set(enriched, '{definition,description}', to_jsonb('Use after writing or editing files to perform a bounded, auditable Python validation in the isolated Sandbox. It is for syntax checks, tests, and a short script whose real exit code is evidence; it is not a general shell, file-editing, network, or package-install tool. command must be python3, args must be a non-empty JSON string array, -c and stdin are forbidden, and timeout_seconds must be 1-30. For a probe or multi-step check, first write a relative script with write_file, then run it.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,command,description}', to_jsonb('Fixed executable: python3 only.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,args,description}', to_jsonb('Required non-empty JSON array; first item is a relative script path, or use -m followed by py_compile, compileall, or unittest. Never use -c, shell syntax, or stdin.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,timeout_seconds,description}', to_jsonb('Optional timeout in seconds, minimum 1 and maximum 30.'::text), true);
            enriched := jsonb_set(enriched, '{definition,input_schema,properties,working_directory,description}', to_jsonb('Optional canonical workspace-relative directory; omit when the workspace root is sufficient.'::text), true);
        END CASE;

        IF NOT EXISTS (
            SELECT 1 FROM agent_platform.tool_versions existing
            WHERE existing.definition_id = item.definition_id
              AND existing.version = item.version + 1
        ) THEN
            INSERT INTO agent_platform.tool_versions (definition_id, version, spec, spec_hash, status, created_by)
            VALUES (item.definition_id, item.version + 1, enriched,
                    'sha256:' || encode(digest(enriched::text, 'sha256'), 'hex'),
                    'published', 'tool-description-v2');
        END IF;

        SELECT jsonb_build_object('id', v.id::text, 'version', v.version::text)
        INTO enriched
        FROM agent_platform.tool_versions v
        WHERE v.definition_id = item.definition_id
          AND v.version = item.version + 1;
        refs := refs || jsonb_build_array(enriched);
    END LOOP;

    SELECT id INTO toolset_definition
    FROM agent_platform.toolset_definitions
    WHERE tenant_id = 'demo' AND toolset_key = 'autonomous-python-tools';

    -- This is a data rollout, not a schema prerequisite. Fresh/test tenants
    -- may not contain the demo seed catalog; in that case the migration must
    -- remain a deterministic no-op instead of inserting a NULL foreign key.
    IF toolset_definition IS NULL OR jsonb_array_length(refs) = 0 THEN
        RETURN;
    END IF;

    toolset_spec := jsonb_build_object('tools', refs);
    IF NOT EXISTS (
        SELECT 1 FROM agent_platform.toolset_versions
        WHERE definition_id = toolset_definition AND version = 9
    ) THEN
        INSERT INTO agent_platform.toolset_versions (definition_id, version, spec, spec_hash, status, created_by)
        VALUES (toolset_definition, 9, toolset_spec,
                'sha256:' || encode(digest(toolset_spec::text, 'sha256'), 'hex'),
                'published', 'tool-description-v2');
    END IF;

    SELECT id INTO toolset_id
    FROM agent_platform.toolset_versions
    WHERE definition_id = toolset_definition AND version = 9;

    SELECT av.agent_id, av.spec INTO target_agent_id, agent_spec
    FROM agent_platform.agent_versions av
    WHERE id = '17d2d18f-7e2f-4c23-8120-194466975d88'::uuid;

    IF target_agent_id IS NULL OR agent_spec IS NULL THEN
        RETURN;
    END IF;
    agent_spec := jsonb_set(agent_spec, '{toolset_ref}', jsonb_build_object('id', toolset_id::text, 'version', '9'), true);

    IF NOT EXISTS (
        SELECT 1 FROM agent_platform.agent_versions
        WHERE agent_id = target_agent_id AND version = 13
    ) THEN
        INSERT INTO agent_platform.agent_versions (agent_id, version, spec, spec_hash, status, created_by, published_at)
        VALUES (target_agent_id, 13, agent_spec,
                'sha256:' || encode(digest(agent_spec::text, 'sha256'), 'hex'),
                'published', 'tool-description-v2', now());
    END IF;

    UPDATE agent_platform.agent_definitions
    SET active_version_id = (SELECT av.id FROM agent_platform.agent_versions av WHERE av.agent_id = target_agent_id AND av.version = 13), updated_at = now()
    WHERE id = target_agent_id;
END $$;

COMMIT;
