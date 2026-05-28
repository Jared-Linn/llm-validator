# Git 工作流规范

> 本仓库采用规范化 Git 工作流，请遵循以下约定。

## 🌿 分支策略

```
main ───── 生产分支，只接受 squash merge
  │
  ├─ feat/xxx      — 新功能
  ├─ fix/xxx       — 修 Bug
  ├─ docs/xxx      — 文档
  ├─ refactor/xxx  — 重构
  ├─ perf/xxx      — 性能优化
  ├─ test/xxx      — 测试
  └─ chore/xxx     — 杂项/配置
```

**规则：**
- 禁止直接向 `main` 提交代码
- 每次修改从 `main` 新建分支 → 开发 → PR → squash merge 回 `main`
- 分支名全小写，用 `/` 分类（如 `feat/add-user-auth`）
- 完成分支后删除远程分支

## 📝 提交信息规范 (Conventional Commits)

```
<type>(<scope>): <简短描述>

<详细说明（可选，72字折行）>
```

| 类型 | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | 修 Bug |
| `docs` | 文档 |
| `refactor` | 重构 |
| `perf` | 性能优化 |
| `test` | 测试 |
| `chore` | 构建/工具/配置 |
| `ci` | CI/CD |

**示例：**
```
feat(auth): add JWT authentication middleware

fix(api): correct redirect URL after login

docs: update README with deployment instructions
```

## 🔄 标准工作流程

```bash
# 1. 从最新的 main 开始
git checkout main && git pull origin main

# 2. 创建功能分支
git checkout -b feat/add-user-auth

# 3. 开发 + 提交（小步提交，原子化）
git add <files>
git commit -m "feat: add user model"

# 4. 推送到远程
git push -u origin HEAD

# 5. 在 GitHub 上创建 Pull Request
#    base: main ← head: feat/add-user-auth

# 6. CI 通过后 squash merge 到 main
#    Merge 方法：Squash and merge

# 7. 本地同步 + 删除分支
git checkout main && git pull origin main
git branch -d feat/add-user-auth
```

## ✅ 提交前检查清单

- [ ] 无硬编码密钥/密码
- [ ] 无 debug print / console.log
- [ ] 无注释代码
- [ ] 新增代码有测试（如果有测试框架）
- [ ] README 已更新（如需要）
- [ ] commit message 符合规范

## 📎 辅助

- 全局 commit 模板已配置：`git commit` 时自动提示格式
- 本项目所有修改请走分支 → PR → merge 流程
