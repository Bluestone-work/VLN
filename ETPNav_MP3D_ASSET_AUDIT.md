# ETPNav MP3D Asset Audit

## Existing assets

| Path | Type | Size | Scene count | Reusable |
| --- | --- | ---: | ---: | --- |
| `/home/wj/VLN-CE/data/scene_datasets/mp3d/2n8kARJN3HM/2n8kARJN3HM.glb` | Extracted GLB | 92 MB | 1 | Yes, partial |
| `/home/wj/VLN-CE/data/scene_datasets/mp3d/YVUC4YcDtcY/YVUC4YcDtcY.glb` | Extracted GLB | 36 MB | 1 | Yes, partial |
| `/home/wj/VLN-CE/data/scene_datasets/mp3d/v1/tasks/mp3d_habitat.zip` | Habitat MP3D archive | 15 GB | 90 GLBs in archive | Yes, after extraction |
| `/home/wj/VLN-CE/data/scene_datasets/mp3d/v1/tasks/extracted/mp3d/` | Extracted Habitat scene directory | ~17 GB | 90 | Yes, direct |

The same two-GLB tree is also visible through `/home/wj/NavQ_ICCV25_official/VLN-CE/data/`; it is a separate existing project tree and was not modified.

## ETPNav target

```text
~/VLN-CE/ETPNav/data/scene_datasets/mp3d/
```

This path is a symlink, so no multi-gigabyte copy was made:

```text
data/scene_datasets/mp3d -> /home/wj/VLN-CE/data/scene_datasets/mp3d/v1/tasks/extracted/mp3d
```

Following the symlink, the target contains 90 scene directories with the expected `{scene}/{scene}.glb` layout.

## Recommended action

The existing authorized Habitat archive was reusable. It has been extracted and connected to ETPNav with a symlink. No new Matterport3D download was started, and the original archive and source directories were preserved.

The two directly visible scenes under the old VLN-CE root were only a partial extraction; they must not be treated as a complete baseline dataset. The extracted Habitat archive provides the full 90-scene set expected by ETPNav.

